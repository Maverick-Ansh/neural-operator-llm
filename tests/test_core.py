"""Correctness tests for the pieces that are easy to get silently wrong.

Every one of these is cheap and runs on CPU. They exist because each failure
mode here produces a model that still *trains* and still *reports a loss* --
just a meaningless one. In particular an FFT convolution that wraps around is
a perfect information leak from the end of the sequence to the beginning, and
would show up only as a suspiciously good validation number.
"""

import sys, os
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nolm.operators import causal_fft_conv, SpectralKernel, SpectralOperatorMixer
from nolm.attention import LocalAttention, FullAttention
from nolm.model import NOLM, NOLMConfig

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# 1. the FFT convolution is the convolution we think it is
# --------------------------------------------------------------------------- #
def direct_causal_conv(v, k):
    """Literal O(N*L) reference implementation of the sum in eq. (2)."""
    B, C, N = v.shape
    L = k.shape[-1]
    out = torch.zeros_like(v)
    for i in range(N):
        for m in range(min(i + 1, L)):
            out[:, :, i] += k[:, m] * v[:, :, i - m]
    return out


def test_fft_conv_matches_direct():
    v = torch.randn(2, 3, 64, dtype=torch.float64)
    k = torch.randn(3, 64, dtype=torch.float64)
    got = causal_fft_conv(v, k)
    want = direct_causal_conv(v, k)
    assert torch.allclose(got, want, atol=1e-9), (got - want).abs().max()


def test_fft_conv_is_not_circular():
    """A circular convolution would let the tail wrap into the head.

    Impulse at the *last* position; a linear causal conv must leave every
    earlier output at exactly zero.
    """
    N = 64
    v = torch.zeros(1, 1, N, dtype=torch.float64)
    v[0, 0, -1] = 1.0
    k = torch.randn(1, N, dtype=torch.float64)
    out = causal_fft_conv(v, k)
    assert out[0, 0, :-1].abs().max() < 1e-12, "convolution wrapped around"
    assert torch.allclose(out[0, 0, -1], k[0, 0])


# --------------------------------------------------------------------------- #
# 2. causality of the whole model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant,op_mode", [
    ("nolm", "stretch"), ("nolm", "fixed"), ("transformer", "stretch"), ("local", "stretch")])
def test_model_is_causal(variant, op_mode):
    """Perturb the last token; every earlier logit must be bit-identical.

    This is the test that actually protects the perplexity numbers.
    """
    cfg = NOLMConfig(d_model=64, n_layers=4, n_heads=4, train_len=32,
                     window=8, n_modes=8, variant=variant, op_mode=op_mode)
    m = NOLM(cfg).double().eval()
    idx = torch.randint(0, 256, (2, 32))

    with torch.no_grad():
        a, _ = m(idx)
        idx2 = idx.clone()
        idx2[:, -1] = (idx2[:, -1] + 7) % 256          # change only the final token
        b, _ = m(idx2)

    delta = (a - b).abs()
    assert delta[:, :-1].max() < 1e-10, f"future leaked into the past: {delta[:, :-1].max()}"
    assert delta[:, -1].max() > 0, "the perturbation had no effect at all -- test is vacuous"


@pytest.mark.parametrize("mode", ["stretch", "fixed"])
def test_operator_layer_causality_gradient(mode):
    """Independent check via autograd: d out[i] / d in[j] must be 0 for j > i.

    Both addressing modes are checked: `fixed` truncates and zero-pads the
    kernel, which is a separate code path and a separate chance to leak.
    """
    op = SpectralOperatorMixer(16, n_modes=8, mode=mode, ref_len=32).double()
    x = torch.randn(1, 24, 16, dtype=torch.float64, requires_grad=True)
    y = op(x)
    y[0, 10].sum().backward()
    assert x.grad[0, 11:].abs().max() < 1e-12, "operator output depends on future inputs"
    assert x.grad[0, :11].abs().max() > 0


# --------------------------------------------------------------------------- #
# 3. discretisation invariance -- the central claim
# --------------------------------------------------------------------------- #
def test_kernel_is_resolution_invariant():
    """The kernel at N and at 4N must be the *same continuous function*.

    The spectral parameterisation means kernel(4N) sampled every 4th point is
    kernel(N), up to the 1/N quadrature weight that torch's irfft folds in.
    """
    sk = SpectralKernel(8, n_modes=16, mode="stretch", learn_decay=True).double()
    N = 128
    k_lo = sk.kernel(N, "cpu", dtype=torch.float64)          # (8, N)
    k_hi = sk.kernel(4 * N, "cpu", dtype=torch.float64)      # (8, 4N)

    # Undo the 1/grid normalisation to recover kappa itself, then subsample.
    kappa_lo = k_lo * N
    kappa_hi = (k_hi * (4 * N))[:, ::4]
    rel = (kappa_lo - kappa_hi).abs().max() / kappa_lo.abs().max()
    assert rel < 1e-9, f"kernel is not resolution invariant, rel err {rel}"


def test_fixed_mode_is_not_resolution_invariant():
    """The control must behave differently -- otherwise the ablation is empty."""
    sk = SpectralKernel(8, n_modes=16, mode="fixed", ref_len=128).double()
    k_lo = sk.kernel(128, "cpu", dtype=torch.float64)
    k_hi = sk.kernel(512, "cpu", dtype=torch.float64)
    assert k_hi.shape[-1] == 512
    # fixed mode keeps absolute lags: the first 128 taps are unchanged...
    assert torch.allclose(k_lo, k_hi[:, :128])
    # ...and there is nothing beyond the reference length.
    assert k_hi[:, 128:].abs().max() == 0


def test_param_count_independent_of_length():
    """A model defined for 2k tokens must have identical parameters at 64k."""
    cfg = NOLMConfig(d_model=64, n_layers=4, n_heads=4, train_len=2048,
                     window=8, n_modes=8, variant="nolm")
    m = NOLM(cfg).eval()
    before = m.num_params()
    with torch.no_grad():
        m(torch.randint(0, 256, (1, 64)))
        m(torch.randint(0, 256, (1, 512)))       # 8x longer, never seen
    assert m.num_params() == before


# --------------------------------------------------------------------------- #
# 4. local attention really is a sliding window
# --------------------------------------------------------------------------- #
def test_local_attention_matches_masked_reference():
    """Compare against dense attention with an explicit band mask."""
    torch.manual_seed(1)
    D, H, W, N = 32, 4, 8, 40
    la = LocalAttention(D, H, window=W).double().eval()
    x = torch.randn(2, N, D, dtype=torch.float64)

    with torch.no_grad():
        got = la(x)

        # reference: same projections, dense scores, band mask
        q, k, v = la.qkv(x).view(2, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
        q, k = la.rotary(q, k)
        scores = (q @ k.transpose(-1, -2)) / (D // H) ** 0.5
        i = torch.arange(N).view(N, 1)
        j = torch.arange(N).view(1, N)
        allowed = (j <= i) & (i - j < W)
        scores = scores.masked_fill(~allowed, float("-inf"))
        o = torch.softmax(scores, -1) @ v
        want = la.out_proj(o.transpose(1, 2).reshape(2, N, D))

    err = (got - want).abs().max()
    assert err < 1e-8, f"local attention != windowed reference, max err {err}"


def test_local_attention_ignores_distant_tokens():
    """Editing a token more than W back must not change the current output."""
    D, H, W, N = 32, 4, 8, 64
    la = LocalAttention(D, H, window=W).double().eval()
    x = torch.randn(1, N, D, dtype=torch.float64)
    with torch.no_grad():
        a = la(x)
        x2 = x.clone()
        x2[0, 0] += 5.0                       # far outside the window of token 63
        b = la(x2)
    assert (a[0, -1] - b[0, -1]).abs().max() < 1e-10


# --------------------------------------------------------------------------- #
# 5. the model actually runs at a length it was never configured for
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("variant,op_mode", [
    ("nolm", "stretch"), ("nolm", "fixed"), ("local", "stretch")])
def test_forward_at_unseen_length(variant, op_mode):
    cfg = NOLMConfig(d_model=64, n_layers=4, n_heads=4, train_len=64,
                     window=16, n_modes=8, variant=variant, op_mode=op_mode)
    m = NOLM(cfg).eval()
    with torch.no_grad():
        for n in (64, 256, 1024):
            logits, loss = m(torch.randint(0, 256, (1, n)),
                             torch.randint(0, 256, (1, n)))
            assert logits.shape == (1, n, 256)
            assert torch.isfinite(loss)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
