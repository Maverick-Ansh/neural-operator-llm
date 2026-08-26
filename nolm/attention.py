"""Attention layers: bounded-window local attention, and a full-attention baseline.

Why two kinds
-------------
The neural operator of ``operators.py`` is a *linear*, band-limited, global
mixer. Band-limiting is exactly what makes it resolution invariant, but it also
means it cannot represent a sharp, content-dependent lookup between two nearby
tokens. Attention can. So the hybrid model pairs them:

    local attention  -> sharp, content-dependent, but only within W tokens
    spectral operator-> global and length-agnostic, but smooth and linear

A second, important consequence: because the local attention window is bounded
by W, the *relative* distances its RoPE ever sees are always in [0, W). They
therefore never go out of distribution when the context grows, and the local
path length-generalises for free. All of the model's ability to fail at long
context is concentrated in the global path -- which is exactly the thing we
want to measure.

The full-attention module is the control: same parameter budget, unbounded
relative distances, and O(N^2) cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# rotary position embedding
# --------------------------------------------------------------------------- #
class Rotary(nn.Module):
    """Standard RoPE (Su et al., 2021), with a cache keyed on sequence length."""

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cache_len, self._cos, self._sin = 0, None, None

    def _build(self, n, device):
        if self._cos is not None and self._cache_len >= n and self._cos.device == device:
            return self._cos[:n], self._sin[:n]
        t = torch.arange(n, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        self._cos, self._sin, self._cache_len = freqs.cos(), freqs.sin(), n
        return self._cos, self._sin

    def forward(self, q, k, offset=0):
        """q, k: (B, H, N, D). Rotates both by their absolute position + offset."""
        n = q.shape[-2]
        cos, sin = self._build(n + offset, q.device)
        cos, sin = cos[offset:offset + n], sin[offset:offset + n]
        cos, sin = cos[None, None], sin[None, None]                 # (1,1,N,D/2)

        def rot(x):
            x1, x2 = x.float().chunk(2, dim=-1)
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)

        return rot(q), rot(k)


# --------------------------------------------------------------------------- #
# bounded sliding-window attention, O(N * W)
# --------------------------------------------------------------------------- #
class LocalAttention(nn.Module):
    """Causal attention restricted to the previous ``window`` tokens.

    Implemented by blocking: the sequence is cut into chunks of ``window``
    tokens and each query chunk attends to its own chunk plus the one before
    it (``2*window`` keys). Every query then sees exactly ``window`` valid
    positions, so this is a true sliding window, at O(N*W) cost and memory
    instead of O(N^2).
    """

    def __init__(self, d_model, n_heads, window=128, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim, self.window = n_heads, d_model // n_heads, window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim)
        self.dropout = dropout
        self._mask_key, self._mask = None, None

    def _get_mask(self, n_chunks, W, device):
        """(n_chunks, W, 2W) bool: True where attention is allowed."""
        key = (n_chunks, W, device)
        if self._mask_key == key:
            return self._mask
        i = torch.arange(W, device=device).view(1, W, 1)
        j = torch.arange(2 * W, device=device).view(1, 1, 2 * W)
        # query local index i sits at global c*W+i; key local index j sits at
        # global (c-1)*W+j. Causal (j <= i globally) and within-window
        # (distance < W) reduce to the single band  i+1 <= j <= W+i.
        m = (j >= i + 1) & (j <= W + i)
        m = m.expand(n_chunks, W, 2 * W).clone()
        # Chunk 0 has no predecessor: its first W keys are zero padding.
        m[0, :, :W] = False
        self._mask_key, self._mask = key, m
        return m

    def forward(self, x):
        B, N, D = x.shape
        W = min(self.window, N)
        pad = (-N) % W                              # right-pad up to a multiple of W
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Np = x.shape[1]
        nc = Np // W

        q, k, v = self.qkv(x).view(B, Np, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.rotary(q, k)                    # (B, H, Np, Dh)

        # Left-pad keys/values by one window so chunk c can look back.
        kp = F.pad(k, (0, 0, W, 0))                 # (B, H, Np+W, Dh)
        vp = F.pad(v, (0, 0, W, 0))
        # unfold -> (B, H, nc, Dh, 2W); move the window axis before the head dim.
        kw = kp.unfold(2, 2 * W, W).permute(0, 1, 2, 4, 3)   # (B,H,nc,2W,Dh)
        vw = vp.unfold(2, 2 * W, W).permute(0, 1, 2, 4, 3)

        qw = q.view(B, self.n_heads, nc, W, self.head_dim)
        mask = self._get_mask(nc, W, x.device).view(1, 1, nc, W, 2 * W)

        o = F.scaled_dot_product_attention(
            qw, kw.contiguous(), vw.contiguous(), attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0)

        o = o.reshape(B, self.n_heads, Np, self.head_dim).transpose(1, 2).reshape(B, Np, D)
        return self.out_proj(o[:, :N])


# --------------------------------------------------------------------------- #
# full causal attention -- the O(N^2) control
# --------------------------------------------------------------------------- #
class FullAttention(nn.Module):
    """Ordinary causal self-attention with RoPE, via memory-efficient SDPA.

    SDPA's flash backend keeps *memory* linear in N, so this can be evaluated at
    long context; the *compute* is still quadratic, which is the point.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, rope_base=10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.dropout = dropout

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.rotary(q, k)
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(o.transpose(1, 2).reshape(B, N, D))
