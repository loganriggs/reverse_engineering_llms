# Finite causality of fixed-rank weight/context route hybrids

At four prospectively selected depth regimes, known semantic modes are retained and the remaining native destination routes are projected into five rank-four allocations. `k=0` is four context/CE-fitted coordinates; `k=4` is the static weight basis. The target-layer MLP and full suffix recompute on fresh FineWeb.

## Locked hypotheses

- **F0_native_path_is_exact: supported.** Observed: `{'maximum_logit_error': 3.999471664428711e-05}`
- **F1_context_rank4_is_causally_strong: supported.** Observed: `{'mean': 0.8604638113880629}`
- **F2_route_geometry_predicts_finite_causality: not supported.** Observed: `{'correlation': -0.3030288825646893, 'cells': 20}`
- **F3_L15_is_a_static_weight_regime: supported.** Observed: `{'context_minus_weight': -0.0020290324658891468}`
- **F4_L11_needs_only_one_context_coordinate: supported.** Observed: `{'context_minus_hybrid': -0.01640917417345311}`
- **F5_early_layers_need_context_conditioning: not supported.** Observed: `{'gaps': {1: -0.019359101568165782, 4: 0.05488044134775194}}`
- **F6_two_context_coordinates_improve_the_weight_prior: not supported.** Observed: `{'layers': [4, 11]}`

| L | Full-logit fidelity k=0/1/2/3/4 | CE-effect fidelity k=0/1/2/3/4 |
|---:|---:|---:|
| 1 | 0.967/0.979/0.986/0.990/0.986 | 0.981/0.990/0.992/0.992/0.987 |
| 4 | 0.869/0.860/0.880/0.875/0.814 | 0.828/0.842/0.863/0.887/0.847 |
| 11 | 0.764/0.743/0.777/0.780/0.773 | 0.912/0.898/0.922/0.920/0.926 |
| 15 | 0.843/0.841/0.840/0.844/0.845 | 0.840/0.834/0.826/0.834/0.814 |

## Guardrail

These context directions are fixed across the new panel; this validates a corpus/loss-conditioned chart, not a per-example dynamic basis. The selected layers represent regimes discovered earlier, so the layerwise claims are prospective for text but not an unbiased estimate over depth.
