# Counterfactual red team of boundary dialect transport

One US/UK spelling cue is swapped in an otherwise identical prefix. The tested output contrast uses four disjoint spelling pairs, rotated over three folds. L9H8's latest-boundary edge is compared with a frequency-near non-boundary edge from the same distance bin and with L9H7 at the same boundary. Thirty-one balanced random signings of the target tokens test whether the declared contrast was privileged.

## Locked results

- **S0_native_model_responds_to_spelling_swap: supported.**
- **S1_boundary_leading_mode_responds: supported.**
- **S2_boundary_edge_transmits_disjoint_style: supported.**
- **S3_transport_is_boundary_specific: supported.**
- **S4_transport_is_head_specific: supported.**
- **S5_declared_contrast_beats_random: supported.**
- **S6_counterfactual_dialect_transport_is_supported: supported.**

| Fold/split | Program | N | leading win | native disjoint win | edge-delta win / mean | contrast percentile |
|---|---|---:|---:|---:|---:|---:|
| 0/0 | L9H8_boundary | 160 | 0.806 | 0.975 | 0.775 / +0.0442 | 1.000 |
| 0/0 | L9H8_nonboundary | 160 | 0.606 | 0.975 | 0.575 / +0.001706 | 1.000 |
| 0/0 | L9H7_boundary | 160 | 0.431 | 0.975 | 0.369 / -0.0008654 | 0.000 |
| 0/1 | L9H8_boundary | 160 | 0.713 | 0.963 | 0.706 / +0.03982 | 1.000 |
| 0/1 | L9H8_nonboundary | 160 | 0.594 | 0.963 | 0.525 / +0.002483 | 1.000 |
| 0/1 | L9H7_boundary | 160 | 0.487 | 0.963 | 0.438 / -0.0005517 | 0.000 |
| 1/0 | L9H8_boundary | 142 | 0.754 | 0.958 | 0.725 / +0.04751 | 1.000 |
| 1/0 | L9H8_nonboundary | 142 | 0.627 | 0.958 | 0.620 / +0.0002004 | 0.806 |
| 1/0 | L9H7_boundary | 142 | 0.493 | 0.958 | 0.408 / -0.0006616 | 0.000 |
| 1/1 | L9H8_boundary | 158 | 0.715 | 0.981 | 0.671 / +0.0429 | 1.000 |
| 1/1 | L9H8_nonboundary | 158 | 0.557 | 0.981 | 0.513 / +8.259e-05 | 1.000 |
| 1/1 | L9H7_boundary | 158 | 0.475 | 0.981 | 0.342 / -0.0007372 | 0.000 |
| 2/0 | L9H8_boundary | 160 | 0.769 | 0.975 | 0.738 / +0.05835 | 1.000 |
| 2/0 | L9H8_nonboundary | 160 | 0.650 | 0.975 | 0.600 / +0.003235 | 1.000 |
| 2/0 | L9H7_boundary | 160 | 0.456 | 0.975 | 0.287 / -0.001151 | 0.000 |
| 2/1 | L9H8_boundary | 160 | 0.731 | 0.950 | 0.738 / +0.0421 | 1.000 |
| 2/1 | L9H8_nonboundary | 160 | 0.619 | 0.950 | 0.562 / +0.001191 | 1.000 |
| 2/1 | L9H7_boundary | 160 | 0.450 | 0.950 | 0.350 / -0.0005386 | 0.000 |

## Guardrails

A one-token swap is a much stronger control for topic than observational labels, but it can be locally unnatural and still perturb generic model state. Disjoint cue and target pairs rule out copying the same spelling token; they do not eliminate a broader lexical-style association learned from correlated domains. The non-boundary control is matched by log-distance and approximate token frequency, not by source residual content. This supports a transportable spelling/style variable, not the claim that L9H8 represents an abstract sociolinguistic concept in every context.
