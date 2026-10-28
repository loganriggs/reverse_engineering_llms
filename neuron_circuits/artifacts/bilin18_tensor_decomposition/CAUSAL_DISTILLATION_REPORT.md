# Causal CP distillation results

## Outcome

- **H1_kl_behavior: not supported.**
- **H2_causal_training: not supported.**
- **H3_tensor_anchor: not supported.**
- **H4_scale_transfer: not supported.**
- **H5_causal_gate: not supported.**

![Causal distillation summary](causal_distillation_summary.png)

## Behavior

| Model | FineWeb KL | WikiText KL | TinyStories KL |
|---|---:|---:|---:|
| Activation CP baseline | 0.0174 | 0.0231 | 0.0118 |
| KL distilled | 0.0192 | 0.0280 | 0.0128 |
| Causally distilled | 0.0193 | 0.0263 | 0.0127 |
| Causal + tensor anchor | 0.0194 | 0.0267 | 0.0134 |

## Causal fidelity

| Model | Gaussian .10 fidelity | Gaussian cosine | Natural .25 fidelity | Natural cosine |
|---|---:|---:|---:|---:|
| Activation CP baseline | 0.157 | 0.557 | 0.584 | 0.807 |
| KL distilled | 0.080 | 0.535 | 0.554 | 0.789 |
| Causally distilled | 0.095 | 0.534 | 0.546 | 0.790 |
| Causal + tensor anchor | 0.093 | 0.536 | 0.548 | 0.788 |

## Global/local tradeoff

| Model | Tensor fidelity | FineWeb write fidelity | Factor cosine to baseline | Activation CKA |
|---|---:|---:|---:|---:|
| Activation CP baseline | 0.363 | 0.870 | 1.000 | 1.000 |
| KL distilled | 0.362 | 0.817 | 1.000 | 0.995 |
| Causally distilled | 0.363 | 0.820 | 1.000 | 0.995 |
| Causal + tensor anchor | 0.362 | 0.822 | 1.000 | 0.995 |

## Red-team interpretation

- Training on a causal metric can improve that metric without proving that the learned factors are canonical or human-semantic.
- Held-out magnitudes and OOD corpora test interpolation and distribution transfer, but not unrelated intervention families or other layers.
- All variants share one architecture and initialization. This is an objective comparison, not independent rediscovery.
- The teacher itself defines the distillation target; agreement does not explain why the teacher computes the behavior.

Raw results: `causal_distillation_results.json`.
