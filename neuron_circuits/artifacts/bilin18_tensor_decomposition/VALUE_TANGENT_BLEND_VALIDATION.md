# Value–tangent blend disjoint validation

## Outcome

- **H1_behavior: supported.**
- **H2_causal: supported.**
- **H3_interior: supported.**

## Behavior

| Corpus | λ=0 KL | λ=.20 KL | λ=.40 KL |
|---|---:|---:|---:|
| fineweb | 0.0144 | 0.0132 | 0.0141 |
| wikitext103 | 0.0189 | 0.0178 | 0.0192 |
| tinystories | 0.0100 | 0.0090 | 0.0092 |

## Causal fidelity

| Corpus/intervention | λ=0 | λ=.20 | λ=.40 |
|---|---:|---:|---:|
| fineweb/gaussian_rms_0.10 | 0.173 | 0.185 | 0.179 |
| fineweb/natural_mix_0.25 | 0.628 | 0.644 | 0.617 |
| wikitext103/gaussian_rms_0.10 | 0.203 | 0.223 | 0.223 |
| wikitext103/natural_mix_0.25 | 0.633 | 0.649 | 0.622 |
| tinystories/gaussian_rms_0.10 | 0.099 | 0.130 | 0.134 |
| tinystories/natural_mix_0.25 | 0.617 | 0.623 | 0.594 |

The λ=.20 coefficient and row ranges were frozen before this run. The blend remains a rank-4096 diagnostic unless recompressed.

Raw results: `value_tangent_blend_validation.json`.
