### Paired comparison of bits/byte (same windows, differenced)

Every model is evaluated on the identical, deterministic sequence of windows, so
the per-window losses pair one-to-one. Differencing them cancels the
between-window variance — which is far larger than the between-model variance —
turning a comparison of two noisy curves into a quantitative one.

Negative favours the first model. `t` is a paired t-statistic over the shared
windows; `|t| > 2` is the usual bar for significance. Computed on the full
per-window losses (`results/training_logs`, `nolm.plots`).

**NOLM (operator, stretch)  minus  Transformer (RoPE, full attn)** — _does the operator beat full attention?_

| length | delta bpb | +/- stderr | t | n |
|---|---|---|---|---|
| 1K | +0.0462 | 0.0051 | +9.0 | 128 |
| 2K | +0.0469 | 0.0049 | +9.5 | 128 |
| 4K | −1.7544 | 0.0226 | −77.5 | 128 |
| 8K | −2.6369 | 0.0325 | −81.1 | 128 |
| 16K | −3.2223 | 0.0660 | −48.8 | 64 |
| 32K | −3.4472 | 0.1096 | −31.4 | 32 |
| 64K | −3.5001 | 0.0964 | −36.3 | 32 |

_In distribution the transformer is better by 0.047 bpb, and the margin is
unambiguous (t ≈ +9). One doubling later the sign flips and the magnitude grows
by a factor of forty._

**NOLM (operator, stretch)  minus  Local attention only** — _does the global operator path do anything at all?_

| length | delta bpb | +/- stderr | t | n |
|---|---|---|---|---|
| 1K | −0.0082 | 0.0034 | −2.4 | 128 |
| 2K | −0.0093 | 0.0031 | −3.0 | 128 |
| 4K | −0.0134 | 0.0040 | −3.3 | 128 |
| 8K | −0.0094 | 0.0043 | −2.2 | 128 |
| 16K | −0.0075 | 0.0056 | −1.3 | 64 |
| 32K | −0.0129 | 0.0081 | −1.6 | 32 |
| 64K | −0.0063 | 0.0080 | −0.8 | 32 |

_This is the decisive control. The operator helps, significantly but by about
0.01 bpb — and the advantage does **not** grow with context. It is largest at
4K and statistically gone by 64K._

**NOLM (operator, stretch)  minus  NOLM (operator, fixed)** — _is resolution-invariant addressing the active ingredient?_

| length | delta bpb | +/- stderr | t | n |
|---|---|---|---|---|
| 1K | +0.0026 | 0.0003 | +8.3 | 128 |
| 2K | +0.0000 | 0.0000 | n/a | 128 |
| 4K | +0.0001 | 0.0004 | +0.3 | 128 |
| 8K | +0.0016 | 0.0007 | +2.4 | 128 |
| 16K | +0.0034 | 0.0012 | +2.8 | 64 |
| 32K | +0.0023 | 0.0013 | +1.8 | 32 |
| 64K | +0.0079 | 0.0019 | +4.3 | 32 |

_At 2K the difference is exactly zero, because at N = ref_len the two addressing
modes are the same computation. Everywhere else the sign is positive: the
resolution-invariant addressing is consistently **worse** than plain absolute
lags, and the gap widens with length (t = +4.3 at 64K)._

**NOLM (operator, stretch)  minus  NOLM weights re-addressed as fixed** — _same weights, re-addressed onto absolute lags_

| length | delta bpb | +/- stderr | t | n |
|---|---|---|---|---|
| 1K | +0.0026 | 0.0003 | +8.3 | 128 |
| 2K | +0.0000 | 0.0000 | n/a | 128 |
| 4K | +0.0001 | 0.0004 | +0.3 | 128 |
| 8K | +0.0016 | 0.0007 | +2.4 | 128 |
| 16K | +0.0034 | 0.0012 | +2.8 | 64 |
| 32K | +0.0023 | 0.0013 | +1.8 | 32 |
| 64K | +0.0079 | 0.0019 | +4.3 | 32 |

_Identical to the row above, to every digit. That is not a copy-paste error: the
separately trained `nolm_fixed` run turned out to be bit-identical to `nolm`
(see the note on the redundant run in the discussion), so re-addressing one set
of weights and training a second set produce the same model._
