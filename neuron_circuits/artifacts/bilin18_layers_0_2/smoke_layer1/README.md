# Bilin18 layer-1 native MLP discriminator atlas

This is a first-pass input-selectivity audit of all 4,608 native bilinear product channels. Each channel computes `Left_j(x) * Right_j(x)` after attention 1 and RMS normalization. A token-class label is selected on disjoint discovery rows and its absolute-activation lift is evaluated on held-out rows.

## Aggregate

- Discovery tokens: 256; held-out tokens: 128.
- Left/right factors choose the same coarse class for 24.2% of all channels and 29.0% of the more usable half.
- Among the more usable half, median held-out class lift is 1.21; 36.0% exceed 1.5× and 18.4% exceed 2×.

- Their median held-out signed class-to-class gap is 0.42 product standard deviations; 22.7% exceed one standard deviation.

`Lift` is mean absolute product activation on the frozen preferred class divided by its mean over all held-out tokens. `Use score = product standard deviation × Down-column norm`; it is only a first-order availability score, not a causal effect on loss.

## Representative native channels

| unit | Left | Right | signed product separator | held-out gap (SD) | lift | strongest + / − contexts |
|---:|---|---|---|---:|---:|---|
| 624 | auxiliary | auxiliary | number → + / auxiliary → − | 0.02 | 0.14 | + ` through middle America in the[ middle] of the night, they`; ` cool on top of your[ kidneys]? Someone told me this`<br>− `[Did] you know you have two`; `[How] AP reported in all formats` |
| 506 | punctuation | punctuation | determiner → + / punctuation → − | 1.22 | 0.92 | + ` DISH, FLU[BS] IN FIELD IN S`; ` America in the middle of[ the] night, they touched down`<br>− ` Years from now, I[�]�ll remember riding the`; ` chilling out, maxin[�]�, relaxin�` |
| 1535 | number | number | punctuation → + / number → − | 7.97 | 16.82 | + ` chilling out, maxin[�]�, relaxin�`; `’, relaxin[�]� all cool on top`<br>− ` IN FIELD IN S[IX]TH STRAIGHT T`; `[Five] Reasons I Love Boston\n` |
| 2027 | number | number | conjunction → + / number → − | 7.94 | 15.64 | + ` your body, just chilling[ out], maxin’`; ` places hours from any AP[ bureau]. Our closest video journalist`<br>− ` little yellow, nine-[volt]-battery-sized`; ` IN FIELD IN S[IX]TH STRAIGHT T` |

## Red-team limits

- The ten token classes are deliberately coarse and incomplete. Low lift means only that this vocabulary failed, not that a channel is incoherent.
- Selection and top contexts come from discovery rows; only the class lift is held out.
- Layer 1 already includes attention-1 and all earlier blocks, so a channel can separate contextual states rather than current-token identity.
- The signed separator names are frozen on discovery data and evaluated as a standardized class-mean gap on held-out data. They remain coarse descriptions, not exhaustive meanings.
- Activation selectivity is descriptive. Finite channel ablations and cross-context transfer are required before assigning a causal verb.
- Native product channels have bilinear scaling and swap gauges. Interpret Left/Right labels as a pair, not as independently canonical neurons.
