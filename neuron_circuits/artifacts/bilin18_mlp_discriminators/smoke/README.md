# Bilin18 layer-0 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 0 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 128; held-out tokens: 64.
- Left/right factors choose the same coarse class for 23.9% of all channels and 29.6% of the more usable half.
- Among the more usable half, median held-out class lift is 1.19; 37.9% exceed 1.5× and 24.3% exceed 2×.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | product | held-out lift | use score | strongest positive token contexts |
|---:|---|---|---|---:|---:|---|
| 1737 | determiner | determiner | determiner | 2.55 | 3419.68 | ` so blue it melts into[ the] sky, horizon`; ` body, just chilling out[,] maxin’`; ` 8, 2012\nWhen[ the] first serious bout of torn` |
| 2264 | pronoun | determiner | pronoun | 1.24 | 3095.82 | `IXTH STRAIGHT[ T]IGER WIN`; `RAIGHT TIGER[ WIN]`; `ELD IN SIXTH[ ST]RAIGHT TIGER` |
| 739 | number | number | number | 26.73 | 1040.09 | ` have two little yellow,[ nine]-volt-battery`; `[Five] Reasons I Love Boston\n`; `-stricken regionsMarch[ 8], 2012\nWhen the` |
| 2869 | auxiliary | number | number | 20.31 | 784.19 | ` have two little yellow,[ nine]-volt-battery`; `-stricken regionsMarch[ 8], 2012\nWhen the`; ` IN FIELD IN S[IX]TH STRAIGHT T` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 0 already includes attention-0 context, so a channel can separate contextual states rather than current-token identity.
- Absolute activation merges the positive and negative poles for class selection; the JSON retains both signed tails.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
