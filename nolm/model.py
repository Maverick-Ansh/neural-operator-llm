"""The language models.

Three variants share every component except the token mixer, so that any
difference in the results is attributable to the mixer alone:

    "nolm"        alternating LocalAttention / SpectralOperatorMixer  (ours)
    "transformer" FullAttention with RoPE everywhere                 (control)
    "local"       LocalAttention everywhere, no global path          (ablation)

The "local" ablation matters: a bounded-window model also length-generalises
trivially (it simply ignores everything beyond W). Comparing against it is what
separates "the operator actually uses the long context" from "the model is
merely not broken by it".
"""

from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import LocalAttention, FullAttention
from .operators import SpectralOperatorMixer


@dataclass
class NOLMConfig:
    vocab_size: int = 256           # raw bytes -- no tokenizer to lose
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    train_len: int = 2048           # sequence length used during training
    variant: str = "nolm"           # nolm | transformer | local
    window: int = 128               # local-attention window
    n_modes: int = 64               # Fourier modes per operator kernel
    op_mode: str = "stretch"        # stretch (resolution invariant) | fixed
    short_conv: int = 4
    mlp_ratio: float = 8 / 3
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_base: float = 10000.0

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        # Always normalise in fp32: under fp16 autocast the sum of squares of a
        # 512-d vector overflows far too easily.
        f = x.float()
        f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (f * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d, ratio=8 / 3, dropout=0.0):
        super().__init__()
        h = int(d * ratio / 64 + 0.5) * 64          # round to a multiple of 64
        self.w12 = nn.Linear(d, 2 * h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.w3(F.silu(a) * b))


class Block(nn.Module):
    """Pre-norm residual block: x + mixer(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, cfg: NOLMConfig, layer_idx: int):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)

        if cfg.variant == "transformer":
            self.mixer = FullAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.rope_base)
            self.kind = "full"
        elif cfg.variant == "local":
            self.mixer = LocalAttention(cfg.d_model, cfg.n_heads, cfg.window, cfg.dropout)
            self.kind = "local"
        elif cfg.variant == "nolm":
            # Even layers look locally and sharply; odd layers look globally and
            # smoothly. Starting with local means the operator always consumes
            # features that already have local context folded in.
            if layer_idx % 2 == 0:
                self.mixer = LocalAttention(cfg.d_model, cfg.n_heads, cfg.window, cfg.dropout)
                self.kind = "local"
            else:
                self.mixer = SpectralOperatorMixer(
                    cfg.d_model, n_modes=cfg.n_modes, mode=cfg.op_mode,
                    ref_len=cfg.train_len, short_conv=cfg.short_conv, dropout=cfg.dropout)
                self.kind = "operator"
        else:
            raise ValueError(f"unknown variant {cfg.variant!r}")

        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio, cfg.dropout)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class NOLM(nn.Module):
    def __init__(self, cfg: NOLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init)
        # Scale residual-output projections by 1/sqrt(2*L) so the residual stream
        # variance stays O(1) with depth (GPT-2 initialisation).
        import math
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for n, p in self.named_parameters():
            if n.endswith("out_proj.weight") or n.endswith("w3.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 * scale * (cfg.d_model ** 0.0))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)

        if targets is None:
            return self.lm_head(x), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               targets.reshape(-1))
        return logits, loss

    # ---------------- bookkeeping ---------------- #
    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def param_breakdown(self):
        out = {}
        for name, mod in self.named_modules():
            if isinstance(mod, (LocalAttention, FullAttention, SpectralOperatorMixer, SwiGLU)):
                out[type(mod).__name__] = out.get(type(mod).__name__, 0) + \
                    sum(p.numel() for p in mod.parameters())
        return out

    # ---------------- generation ---------------- #
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        """Simple non-cached sampler: recomputes the full forward each step.

        Deliberately not KV-cached -- for the operator layers a streaming
        recurrence would be a separate piece of work, and generation here is a
        qualitative check, not a throughput claim.
        """
        self.eval()
        for _ in range(max_new_tokens):
            crop = idx[:, -self.cfg.train_len:]
            logits, _ = self(crop)
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def build_model(cfg: NOLMConfig) -> NOLM:
    return NOLM(cfg)
