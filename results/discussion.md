## What happened

The headline hypothesis splits cleanly in two, and the two halves get opposite
answers.

**The operator model does survive 32× extrapolation. It does not exploit it.**

At 2,048 bytes — the length everything was trained at — the transformer is
better, by 0.047 bpb, and the paired test leaves no doubt (t = +9.5). One
doubling later it has fallen apart: 1.344 → 3.208 bpb at 4K, and 4.952 by 64K.
For reference, chance on 256 byte values is exactly 8.000 bpb, so the model has
not gone to noise — it has collapsed to roughly what you get from byte
frequencies alone, retaining almost nothing of what it knew at 2K. This is the
familiar RoPE failure: relative distances beyond the training window are simply
out of distribution, and nothing in the architecture makes them well defined.

The operator model has no such cliff. It moves from 1.427 bpb at 1K to 1.452 at
64K — essentially flat across a 64-fold change in context, with the identical
parameters, no fine-tuning, and no positional interpolation. In that narrow
sense the neural-operator construction does exactly what it promised: a model
defined for 2,048 tokens really is already defined for 65,536.

**But flat is not the same as good, and this is where the experiment earns its
controls.**

A bounded-window model is also perfectly flat, for the least interesting reason
available: it ignores everything past its window. So the question is whether the
operator's flatness is of a better kind. Two measurements say it is not.

*The window ablation.* The operator path buys about 0.01 bpb over
window-attention-only — real (t = −3.0 at 2K), but tiny. And the sign of the
trend is wrong: the advantage is largest at 4K (t = −3.3) and has decayed to
nothing by 64K (t = −0.8). If the global path were genuinely putting long
context to work, that gap would *widen* with length. It narrows.

*The copy probe.* This is the decisive one, because it has a working positive
control. Plant a 256-byte passage twice and price the second copy: the
transformer saves **0.639 bits/byte** at a separation of 512 and 0.494 at 1,024.
That is real in-context retrieval, and it proves the instrument can detect
retrieval when retrieval exists. Against that reference, every operator variant
scores between −0.05 and +0.03 bits at every separation tested — indistinguishable
from zero, and indistinguishable from the window-only ablation, which by
construction cannot retrieve anything beyond 128 bytes.

The operator does no long-range retrieval at all.

## Why, and why it was predictable

A spectral kernel is a *linear, band-limited, content-independent* convolution.
Retrieval is content-dependent routing: finding the earlier position whose
content matches the present query. A fixed convolution cannot express that at
any resolution, because the weight applied to position `i − m` depends only on
`m`, never on what is stored there. The multiplicative gate helps — it makes the
output input-dependent — but it scales what the convolution already mixed; it
does not change *where* the mixing reads from. Discretisation invariance is
orthogonal to this and cannot supply it.

So the honest summary is that the property transferred and the capability did
not. What a neural operator gives you is a well-posed way to *evaluate the same
operator at a new resolution*. That is genuinely valuable, and it is why the
model does not break. It is not a mechanism for using more information.

## Resolution invariance is the wrong prior for text

The `stretch`/`fixed` ablation is the sharpest result here, because both modes
share the same parameterisation and differ only in the grid the kernel is
resampled onto — so the comparison has no run-to-run confound at all.

Absolute-lag addressing wins, consistently, and by more as context grows
(t = +2.4 at 8K, +4.3 at 64K). The effect is small in absolute terms (0.008 bpb)
but its direction is unambiguous.

That is the expected sign once stated plainly. An FNO earns its keep on PDEs
because the solution is a smooth field on a *fixed domain*: refining the grid
should reveal the same function in more detail, so normalised coordinates are
the physically correct addressing. Text has no fixed domain. Its structure lives
at absolute token distances — a word is four characters from its neighbour
whether the document is 2KB or 200KB. Addressing the kernel in normalised
coordinates means that at 64K context a lag of `t = 0.005` refers to 320 bytes
rather than 10, so every scale the operator learned during training is stretched
by 32× and lands on the wrong structure. The measurement agrees.

## Efficiency, stated with its cost

The `O(N log N)` claim holds in wall-clock. At 64K the operator model runs a
forward pass in 1.16 s against the transformer's 5.87 s — **5.1× faster** — and
the gap widens monotonically with length (at 2K the transformer is slightly
*faster*, 0.022 s vs 0.030 s).

Memory goes the other way, and the README should not pretend otherwise: at 64K
the operator model peaks at 3.53 GB against the transformer's 1.61 GB, **2.2×
worse**. That is an implementation property, not a theoretical one. The FFT path
materialises fp32 buffers of length 2N (cuFFT has no half-precision path worth
relying on), while PyTorch's memory-efficient attention kernel never
materialises the score matrix at all. A chunked overlap-save convolution would
remove most of this; it was not implemented.

## Two things worth recording about the process

**One training run was redundant, and it was derivable in advance.** The
separately trained `nolm_fixed` turned out bit-identical to `nolm` — same
validation number at every checkpoint, same final 1.4083. That is forced: the
two addressing modes coincide exactly when `N = ref_len = train_len`, so
training at 2,048 cannot distinguish them. The distinction only exists at
evaluation. Roughly 67 minutes of GPU went into rediscovering an identity that
five minutes of thought would have produced. It does at least serve as a clean
determinism check, and the re-addressing experiment (same weights, evaluated
both ways) was the right design all along.

**The instrument was bracketed before it was trusted.** A randomly initialised
model reads 7.945 bpb where chance is exactly 8.000, and scores zero copy gain —
that is the floor. The transformer's +0.639 bits at 512 is the ceiling-side
positive control. Both were measured, not assumed, which is what licenses
reading the operator's zero as a real zero rather than a broken probe.

Related: in the length sweep the *unpaired* error bars overlap almost completely
(see `fig_bpb_tail_zoom.png`) — between-window variance in enwik8 is far larger
than any difference between these models. Only the paired test, differencing the
same windows model-by-model, resolves the comparison. Reporting the unpaired
curves alone would have supported no conclusion in either direction.

## Limitations

These results are one seed, one scale (~30M parameters), one corpus, and about
105M bytes of training — roughly a single pass over enwik8. The absolute
numbers (1.34–1.45 bpb) are well short of what enwik8 models reach with a full
schedule, so all of this describes the small-model, short-training regime.
Learning rate was fixed at 1e-3 for every architecture with no per-architecture
tuning; that choice cannot explain the length-generalisation result, which is
about the shape of the curve, but it does leave the in-distribution comparison
less settled than it looks. The operator used 64 Fourier modes throughout, and
mode count was never swept.

## What would actually test the idea next

The finding is not that spectral mixing is useless — it is that a
content-independent kernel cannot retrieve, and resolution invariance does not
change that. The interesting follow-ups all attack that specific gap:

1. **Give the global path content-dependent addressing.** A small number of
   global attention heads alongside the operator, or a kernel whose coefficients
   are predicted from a pooled summary of the sequence. If the copy probe moves
   off zero, the diagnosis here is confirmed.
2. **Train at long context.** Every model here was trained at 2K, so none had any
   gradient signal rewarding long-range structure. The operator may simply never
   have been asked. Training at 8K–16K and re-running the same sweep separates
   "cannot" from "was never taught to".
3. **Sweep the mode count.** 64 modes over a whole document is a very smooth
   kernel. Whether more modes buy anything, or just cost compute, is unmeasured.
