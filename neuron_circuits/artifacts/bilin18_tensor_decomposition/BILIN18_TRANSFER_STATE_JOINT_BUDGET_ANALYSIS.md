# Runtime-coordinate budget for joint and collapsed charts

A collapsed rank-r residual chart supplies r coefficients independently at each of 128 token positions, for 128r runtime coordinates. A joint rank-r sequence-field chart supplies only r coordinates total. This post-same-rank analysis compares causal fidelity under that execution budget. Chart-generation and declaration costs are discussed separately.

## Locked hypotheses

- **B1_joint64_beats_more_expensive_collapsed1: supported.**
- **B2_joint128_beats_coordinate_matched_collapsed1: supported.**
- **B3_joint64_beats_collapsed4_despite_eighth_coordinates: supported.**
- **B4_joint64_is_useful_at_early_boundary: supported.**
- **B5_joint_frontier_is_coordinate_pareto_superior: supported.**

## Coordinate frontier

| Total coordinates | Best joint mean fidelity | Best collapsed mean fidelity | Collapsed ranks available |
|---:|---:|---:|---|
| 1 | 0.094 | — | [] |
| 4 | 0.264 | — | [] |
| 16 | 0.474 | — | [] |
| 64 | 0.687 | — | [] |
| 128 | 0.767 | 0.080 | [1] |

| Target | Boundary | Collapsed r1 (128 coords) | Collapsed r4 (512) | Joint r64 (64) | Joint r128 (128) |
|---|---:|---:|---:|---:|---:|
| L1H6 | 6 | 0.077 | 0.362 | 0.860 | 0.904 |
| L1H6 | 10 | 0.050 | 0.375 | 0.754 | 0.818 |
| L1H6 | 14 | 0.066 | 0.338 | 0.655 | 0.742 |
| L1H6 | 17 | 0.119 | 0.276 | 0.629 | 0.707 |
| L3H4 | 6 | 0.050 | 0.250 | 0.805 | 0.870 |
| L3H4 | 10 | 0.050 | 0.246 | 0.697 | 0.779 |
| L3H4 | 14 | 0.080 | 0.227 | 0.604 | 0.713 |
| L3H4 | 17 | 0.048 | 0.207 | 0.600 | 0.682 |
| L5H7 | 6 | 0.008 | 0.137 | 0.758 | 0.836 |
| L5H7 | 10 | 0.133 | 0.226 | 0.682 | 0.766 |
| L5H7 | 14 | 0.155 | 0.261 | 0.611 | 0.709 |
| L5H7 | 17 | 0.123 | 0.211 | 0.584 | 0.683 |

## Guardrails

Runtime coordinate count is not total MDL. A literal joint basis contains sequence_length × residual_dimension × rank numbers, while a collapsed basis is much cheaper to store. The joint basis is useful only because it is generated from the tensor-network Fisher program rather than transmitted literally. Compute cost, generator declaration, and residual error must all be included before calling either representation shorter overall.
