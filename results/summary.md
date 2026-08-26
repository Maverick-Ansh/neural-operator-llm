### Bits per byte vs. evaluation context length

| model | params | 1K | 2K | 4K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|---|---|---|
| Local attention only | 31.3M | 1.433 | 1.440 | 1.454 | 1.465 | 1.447 | 1.444 | 1.458 |
| NOLM (operator, stretch) | 30.3M | 1.427 | 1.433 | 1.445 | 1.456 | 1.439 | 1.431 | 1.452 |
| NOLM weights re-addressed as fixed | 30.3M | 1.425 | 1.433 | 1.445 | 1.454 | 1.436 | 1.429 | 1.444 |
| NOLM (operator, fixed) | 30.3M | 1.425 | 1.433 | 1.445 | 1.454 | 1.436 | 1.429 | 1.444 |
| Transformer (RoPE, full attn) | 31.3M | 1.353 | 1.344 | 3.208 | 4.093 | 4.662 | 4.878 | 4.952 |
| NOLM, trained at 8K | 30.3M | 1.489 | 1.490 | 1.500 | 1.508 | 1.490 | 1.480 | 1.508 |
| Transformer, trained at 8K | 31.3M | 1.548 | 1.536 | 1.546 | 1.554 | 4.037 | 4.344 | 4.523 |

_bits/byte over the final 512 bytes of each window; all models trained at 2048._


### Long-range copy gain (bits/byte saved on the second copy)

| model | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|
| Local attention only | +0.015 | -0.013 | -0.001 | +0.008 | +0.018 | -0.018 | +0.004 |
| NOLM (operator, stretch) | +0.026 | -0.011 | -0.018 | -0.022 | -0.014 | -0.036 | -0.005 |
| NOLM weights re-addressed as fixed | +0.029 | -0.011 | -0.027 | -0.011 | -0.005 | -0.047 | -0.007 |
| NOLM (operator, fixed) | +0.029 | -0.011 | -0.027 | -0.011 | -0.005 | -0.047 | -0.007 |
| Transformer (RoPE, full attn) | +0.639 | +0.494 | -0.005 | -2.017 | -2.450 | -2.822 | -3.581 |
| NOLM, trained at 8K | +0.018 | -0.003 | -0.008 | -0.005 | -0.019 | -0.048 | -0.010 |
| Transformer, trained at 8K | +0.382 | +0.194 | +0.006 | -0.040 | -0.048 | -2.210 | -2.826 |

### Cost at long context

| model | 1K s/fwd | 2K s/fwd | 4K s/fwd | 8K s/fwd | 16K s/fwd | 32K s/fwd | 64K s/fwd |
|---|---|---|---|---|---|---|---|
| Local attention only | 0.02 | 0.03 | 0.07 | 0.13 | 0.27 | 0.54 | 1.08 |
| NOLM (operator, stretch) | 0.02 | 0.03 | 0.06 | 0.13 | 0.26 | 0.55 | 1.16 |
| NOLM weights re-addressed as fixed | 0.02 | 0.03 | 0.06 | 0.12 | 0.24 | 0.50 | 1.05 |
| NOLM (operator, fixed) | 0.02 | 0.03 | 0.06 | 0.12 | 0.24 | 0.51 | 1.07 |
| Transformer (RoPE, full attn) | 0.01 | 0.02 | 0.05 | 0.13 | 0.37 | 1.41 | 5.87 |
| NOLM, trained at 8K | 0.02 | 0.03 | 0.06 | 0.11 | 0.23 | 0.48 | 1.05 |
| Transformer, trained at 8K | 0.01 | 0.02 | 0.05 | 0.13 | 0.37 | 1.40 | 5.86 |
