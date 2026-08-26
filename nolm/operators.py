"""
Causal spectral neural operators for autoregressive sequence modelling.

The idea
--------
A neural operator (Li et al., 2020, "Fourier Neural Operator") learns a mapping
between *function spaces* rather than between fixed-dimensional vectors. Its
defining property is **discretisation invariance**: the parameters describe a
continuous kernel, so the same learned operator can be applied at any sampling
resolution.

We reuse that property for language modelling. Treat a length-N token sequence
as a discretisation of a latent signal v : [0, 1] -> R^C sampled on the uniform
grid x_i = i/N. A (causal) kernel-integral operator is

    (K v)(x) = int_0^x  kappa(x - y) v(y) dy                                (1)

whose Riemann sum on the grid is a causal discrete convolution

    u_i = (1/N) sum_{m=0}^{i} kappa(m/N) v_{i-m}.                           (2)

We parameterise kappa by its first K Fourier coefficients R in C^{K x C}. To
run the layer at *any* length N we simply inverse-FFT R onto an N-point grid:
that is band-limited interpolation of the same continuous kernel. Crucially

    the parameter count is O(K*C) and does not depend on N,

so a model trained at N = 2048 is well defined at N = 65536 with *no new
parameters and no retraining*. This is the property we test.

Two addressing modes
--------------------
stretch : kappa is sampled at m/N. The operator is a genuine discretisation of
          eq. (1) on [0,1]; its receptive field is the whole document and it
          "zooms out" as the context grows. This is the true neural operator
          and the resolution-invariant one.

fixed   : kappa is sampled at m/ref_len and zero-padded beyond ref_len. The
          receptive field is a fixed number of *tokens*, matching the usual
          convolutional inductive bias. Not resolution invariant, included as
          the scientific control.

Causality
---------
Note that eq. (2) only ever uses lags m >= 0, so the operator is causal *by
construction* -- there is no mask to get wrong. The usual FNO trick of
multiplying in the frequency domain is NOT used, because that implements a
*circular* convolution which leaks the end of the sequence into the beginning.
We instead materialise the real-space kernel and use a zero-padded FFT, which
computes an exact *linear* convolution in O(N log N).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# exact causal convolution in O(N log N)
# --------------------------------------------------------------------------- #
def causal_fft_conv(v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Linear (non-circular) causal convolution along the last axis.

    Computes  u[b, c, i] = sum_{m=0}^{min(i, L-1)} k[c, m] * v[b, c, i - m].

    Args:
        v: (B, C, N) signal.
        k: (C, L) kernel, k[c, m] is the weight applied at lag m.

    Zero-padding to >= N + L - 1 is what makes this a *linear* convolution; a
    plain length-N FFT product would wrap around and destroy causality.
    """
    N, L = v.shape[-1], k.shape[-1]
    n = 1 << (N + L - 1).bit_length()          # next power of two, fastest cuFFT

    # cuFFT has no half-precision path we can rely on, so half inputs are lifted
    # to fp32; fp64 is preserved so the test-suite can check the maths exactly.
    ct = torch.float32 if v.dtype in (torch.float16, torch.bfloat16) else v.dtype
    vf = torch.fft.rfft(v.to(ct), n=n)
    kf = torch.fft.rfft(k.to(ct), n=n)
    u = torch.fft.irfft(vf * kf.unsqueeze(0), n=n)[..., :N]
    return u.to(v.dtype)


# --------------------------------------------------------------------------- #
# the learned continuous kernel
# --------------------------------------------------------------------------- #
class SpectralKernel(nn.Module):
    """A continuous causal kernel stored as n_modes Fourier coefficients.

    kernel(N) materialises the kernel on an N-point grid. Because the
    parameters are spectral coefficients, this is *resampling one continuous
    function*, not interpolating a discrete array -- which is exactly why the
    module can be evaluated at a length it never saw during training.
    """

    def __init__(self, channels, n_modes=64, mode="stretch", ref_len=2048,
                 init_scale=1e-2, learn_decay=True):
        super().__init__()
        assert mode in ("stretch", "fixed")
        self.channels, self.n_modes = channels, n_modes
        self.mode, self.ref_len = mode, ref_len

        # Real/imag stored separately so the optimiser sees ordinary real tensors.
        # 1/sqrt(mode) scaling gives low modes more energy than high ones, so the
        # kernel starts smooth -- the FNO initialisation.
        decayed = init_scale / torch.sqrt(torch.arange(1, n_modes + 1, dtype=torch.float32))
        self.weight = nn.Parameter(torch.randn(2, n_modes, channels) * decayed[None, :, None])

        # Per-channel exponential envelope exp(-softplus(log_decay) * t). Lets a
        # channel choose to be local even though its kernel is globally supported.
        self.learn_decay = learn_decay
        if learn_decay:
            self.log_decay = nn.Parameter(torch.linspace(-2.0, 3.0, channels))

        # S4-style direct/skip term: the m=0 tap, handled explicitly and unclipped.
        self.D = nn.Parameter(torch.randn(channels) * 0.5)

    def kernel(self, N: int, device, dtype=None) -> torch.Tensor:
        """Materialise the kernel, shape (C, L) with L = N."""
        grid = N if self.mode == "stretch" else self.ref_len
        # Build in the parameter's own precision (fp32 minimum): the spectral
        # round-trip is where resolution invariance lives, so it must not be
        # silently truncated to fp32 when the module is run in fp64.
        rt = self.weight.dtype
        if rt not in (torch.float32, torch.float64):
            rt = torch.float32
        ct = torch.complex128 if rt == torch.float64 else torch.complex64
        if dtype is None:
            dtype = rt

        # Assemble the half-spectrum of a length-`grid` real signal and irfft it.
        # torch's default "backward" normalisation divides by `grid`, which is
        # precisely the 1/N quadrature weight demanded by eq. (2). So the
        # continuous-operator normalisation comes out for free.
        n_freq = grid // 2 + 1
        K = min(self.n_modes, n_freq)
        spec = torch.zeros(n_freq, self.channels, device=device, dtype=ct)
        w = self.weight.to(device=device, dtype=rt)
        spec[:K] = torch.complex(w[0, :K], w[1, :K])

        k = torch.fft.irfft(spec, n=grid, dim=0).transpose(0, 1)   # (C, grid)

        if self.learn_decay:
            t = torch.arange(grid, device=device, dtype=rt) / grid
            lam = F.softplus(self.log_decay.to(device)).unsqueeze(1)   # (C, 1)
            k = k * torch.exp(-lam * t.unsqueeze(0))

        if self.mode == "fixed":
            # Finite support of ref_len tokens: truncate or zero-pad out to N.
            k = k[:, :N] if N <= grid else F.pad(k, (0, N - grid))
        return k.to(dtype)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: (B, C, N) -> (B, C, N)."""
        k = self.kernel(v.shape[-1], v.device)
        return causal_fft_conv(v, k) + v * self.D.to(v.dtype).view(1, -1, 1)

    def extra_repr(self):
        return (f"channels={self.channels}, n_modes={self.n_modes}, "
                f"mode={self.mode}, ref_len={self.ref_len}")


# --------------------------------------------------------------------------- #
# the mixer block
# --------------------------------------------------------------------------- #
class SpectralOperatorMixer(nn.Module):
    """Gated neural-operator token mixer -- the drop-in replacement for attention.

    A bare linear operator is a weak mixer: it cannot express content-dependent
    routing. Following the gated long-convolution family (GSS / Hyena) we make
    the operator multiplicative,

        y = W_o [ (K * short_conv(v)) . silu(g) ],   (v, g) = W_in x

    which restores input-dependence while keeping the O(N log N) cost. The
    short depthwise convolution supplies the sharp local detail that a
    band-limited global kernel provably cannot represent.
    """

    def __init__(self, d_model, n_modes=64, mode="stretch", ref_len=2048,
                 short_conv=4, expand=1.0, dropout=0.0):
        super().__init__()
        d_inner = int(d_model * expand)
        self.d_inner, self.short_conv = d_inner, short_conv

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        # Depthwise, left-padded => causal.
        self.conv = nn.Conv1d(d_inner, d_inner, short_conv, groups=d_inner, bias=True)
        self.op = SpectralKernel(d_inner, n_modes=n_modes, mode=mode, ref_len=ref_len)
        self.norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                    # x: (B, N, D)
        v, g = self.in_proj(x).chunk(2, dim=-1)
        v = v.transpose(1, 2)                                # (B, C, N)
        v = self.conv(F.pad(v, (self.short_conv - 1, 0)))    # causal short conv
        v = self.op(v)
        # Normalise in the norm's own (unautocast) precision, then come back.
        v = self.norm(v.transpose(1, 2).to(self.norm.weight.dtype)).to(x.dtype)
        return self.drop(self.out_proj(v * F.silu(g)))
