# Neural Operator LM

**A byte-level language model whose global token mixer is a Fourier neural operator.
Trained at 2,048 bytes of context. Evaluated at 65,536 — same weights, no retraining,
no fine-tuning, no positional interpolation.**

Everything here is written from scratch in PyTorch: the operator, the attention, the
model, the data pipeline, the training loop, the evaluation. Roughly 30M parameters,
trained on a free Colab T4.

---

## The idea

A **neural operator** ([Li et al., 2020](https://arxiv.org/abs/2010.08895)) learns a
map between *function spaces* rather than between fixed-dimensional vectors. Its
defining property is **discretisation invariance**: the parameters describe a
continuous kernel, so the same learned operator can be applied at any sampling
resolution. Train an FNO on a 64×64 grid, run it on 256×256.

That property is exactly what a long-context language model needs, and exactly what
attention does not have.

Treat a length-`N` token sequence as a discretisation of a latent signal
`v : [0,1] → R^C` sampled on the uniform grid `x_i = i/N`. A causal kernel-integral
operator is

```
(K v)(x) = ∫₀ˣ κ(x − y) · v(y) dy                                          (1)
```

whose Riemann sum on the grid is a causal discrete convolution

```
u_i = (1/N) · Σ_{m=0..i} κ(m/N) · v_{i−m}                                   (2)
```

We parameterise `κ` by its first `K` Fourier coefficients `R ∈ C^{K×C}`. To run the
layer at *any* length `N`, inverse-FFT `R` onto an `N`-point grid — band-limited
interpolation of the same continuous kernel. Two consequences:

1. **The parameter count is `O(K·C)` and does not depend on `N`.** A model defined for
   2,048 tokens is *already* defined for 65,536. There is nothing to extend.
2. **Cost is `O(N log N)`**, not `O(N²)`, because eq. (2) is a convolution.

RoPE attention has neither property: its relative-position function is queried at
distances it never saw in training, and it costs `O(N²)`.

### Causality — the part that is easy to get wrong

The textbook FNO multiplies in the frequency domain. That implements a **circular**
convolution, which wraps the end of the sequence into the beginning — a perfect
information leak from the future, and one that shows up only as a suspiciously good
validation number. We instead materialise the real-space kernel and convolve with a
zero-padded FFT, which computes an exact **linear** convolution in `O(N log N)`.
Because eq. (2) only ever uses lags `m ≥ 0`, the result is causal by construction.

[`tests/test_core.py`](tests/test_core.py) asserts this three ways: against a literal
`O(N²)` reference convolution, against an impulse placed at the final position, and
through autograd (`∂out_i/∂in_j = 0` for `j > i`).

### Two addressing modes

The interesting design question is *what `κ`'s argument means* when the context grows.

| mode | `κ` sampled at | receptive field | resolution invariant |
|---|---|---|---|
| **`stretch`** | `m/N` | the whole document, "zooms out" as `N` grows | **yes** |
| **`fixed`** | `m/ref_len`, zero-padded past `ref_len` | a fixed number of *tokens* | no |

`stretch` is the genuine neural operator — a discretisation of eq. (1) on `[0,1]`.
`fixed` is the ordinary convolutional inductive bias, included as the scientific
control. They share the identical parameterisation and differ only in the grid the
kernel is resampled onto, which makes the comparison unusually clean.

---

## Architecture

A band-limited global kernel is smooth and linear; it provably cannot represent a
sharp, content-dependent lookup between two nearby tokens. Attention can, but only
pays off locally. So the model interleaves them:

```
even layers   LocalAttention          sharp, content-dependent, window W = 128
odd layers    SpectralOperatorMixer   global, length-agnostic, O(N log N)
```

Because the local window is bounded by `W`, the *relative* distances its RoPE ever
sees stay in `[0, W)` — they never go out of distribution when the context grows. All
of the model's ability to fail at long context is therefore concentrated in the global
path, which is exactly the thing being measured.

The operator is gated (`y = W_o[(K * short_conv(v)) ⊙ silu(g)]`), following the gated
long-convolution family (GSS / Hyena), which restores input-dependence that a bare
linear operator lacks. A depthwise short convolution supplies the sharp local detail
the band-limited kernel cannot.

### The three models compared

Everything except the token mixer is shared, so any difference is attributable to the
mixer alone.

| variant | mixer | params |
|---|---|---|
| `nolm` | alternating local attention / spectral operator | **30.3M** |
| `transformer` | full causal attention + RoPE everywhere | 31.3M |
| `local` | local attention everywhere, no global path | 31.3M |

Note NOLM has ~3% **fewer** parameters than its controls, so the comparison is
conservative rather than flattering.

The `local` ablation is the one that matters most. A bounded-window model also
"length-generalises" — trivially, by ignoring everything past `W`. Comparing against
it is what separates *the operator actually uses the long context* from *the model is
merely not broken by it*.

---

## Data

Byte-level [enwik8](http://mattmahoney.net/dc/textdata.html), standard splits
(90M train / 5M validation / 5M test), reported in **bits per byte**.

Bytes rather than BPE, for two load-bearing reasons:

1. **Long context becomes necessary rather than decorative.** A byte is worth roughly
   a quarter of a BPE token, so 64K bytes is only ~16K tokens of ordinary text.
   Structure a subword model would see *inside* its window is pushed outside it.
2. **There is no tokenizer to lose.** The vocabulary is the 256 byte values, fixed by
   the UTF-8 standard rather than by an artefact that must be saved next to the
   checkpoint. Any checkpoint in this repo can be decoded by anyone, forever.

---

## How it is evaluated

Three measurements, in increasing order of how hard they are to fake.

**1. Bits/byte vs. evaluation length.** Reported two ways. `bpb_all` averages over
every position — but early positions in a long window have almost no context no matter
how good the model is, so `bpb_all` is dominated by them and barely moves. `bpb_tail`
averages only the final 512 bytes, i.e. the positions that actually have ~`L` bytes of
context available. **`bpb_tail` falling as `L` grows is the signal; rising means the
long context is actively hurting.**

**2. Long-range copy probe.** A model that merely *tolerates* long context is
indistinguishable from one that *uses* it under (1) — a window model quietly ignoring
everything past `W` has a perfectly flat curve. So we plant a 256-byte passage twice,
separated by a controlled distance, and measure the bits saved on the second copy
relative to the first. The first copy is the control: identical text, identical local
statistics, but nothing to retrieve. Only retrieval across the gap scores.

**3. Cost vs. length.** Seconds per forward pass and peak memory. Where `O(N log N)`
stops being a claim on paper.

Plus a zero-cost controlled experiment: **re-addressing trained weights.** The
spectral coefficients are the same numbers in `stretch` and `fixed` mode — only the
resampling grid changes. Flipping the mode at load time isolates the effect of
resolution-invariant addressing on a fixed set of trained weights, with no retraining
and no run-to-run confound.

---

## Results

<!-- RESULTS -->
_Populated by `nolm.plots` once the evaluation sweep completes._

---

## Reproducing

```bash
git clone https://github.com/Maverick-Ansh/neural-operator-llm
cd neural-operator-llm
pip install -r requirements.txt

python -m pytest tests/test_core.py -q          # 13 correctness tests, CPU, ~3s

python -m nolm.train --variant nolm --op-mode stretch --max-steps 1600 --out runs/nolm
python -m nolm.train --variant transformer      --max-steps 1600 --out runs/transformer
python -m nolm.train --variant local            --max-steps 1600 --out runs/local

python -m nolm.evaluate --ckpt runs/nolm/final.pt --name nolm
python -m nolm.plots
```

enwik8 downloads automatically on first run (~36MB). Every run is checkpointed on a
wall-clock timer and resumable with `--resume`; metrics stream to append-only
`runs/<name>/metrics.jsonl`, so an interrupted run still leaves a complete record.

### Hardware notes

Developed on a free Colab 2×T4 instance. The T4 is compute capability 7.5: it reports
`torch.cuda.is_bf16_supported() == True`, but bf16 there is emulated rather than run
on the tensor cores and is several times slower. **fp16 with a gradient scaler is the
correct choice on this hardware**, and is what `nolm/train.py` uses.

---

## Layout

```
nolm/operators.py   spectral neural operator: kernel, causal FFT conv, gated mixer
nolm/attention.py   bounded sliding-window attention, RoPE, full-attention control
nolm/model.py       the three model variants
nolm/data.py        byte-level enwik8
nolm/train.py       training loop (fp16, wall-clock budgeted, resumable)
nolm/evaluate.py    length sweep, copy probe, cost scaling, kernel snapshots
nolm/plots.py       figures + summary table
tests/test_core.py  causality, non-circularity, resolution invariance, windowing
scripts/            GPU orchestration
```

## References

- Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- Poli et al., *Hyena Hierarchy*, [arXiv:2302.10866](https://arxiv.org/abs/2302.10866)
- Gu et al., *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*, [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

## License

MIT
