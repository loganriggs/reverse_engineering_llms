# Novelty red team for the CE-adjoint residual atlas

The CE-selected residual subspace can recur simply because ordinary route covariance recurs. This report compares it with the earlier task-agnostic high-magnitude route basis, then removes that generic rank-four space and tests rank-two complement recurrence against per-edge head permutations.

## Locked hypotheses

- **N0_ce_selection_adds_predictive_geometry: not supported.** Observed: `{'layers': [1, 2, 3, 4, 5, 6, 7, 10, 13, 14]}`
- **N1_top_residual_routes_are_not_just_gradients: supported.** Observed: `{'layers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}`
- **N2_both_helpful_and_harmful_poles_are_present: supported.** Observed: `{'layers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}`
- **N3_novel_complement_recurrence_beats_head_permutation: supported.** Observed: `{'observed': 0.8649979118715906, 'null_mean': 0.6481393895952925, 'null_p95': 0.6970166786337336}`

| L | CE/generic fidelity | Advantage | CE↔generic basis | Novel complement cosine | Route↔gradient D/R | Positive pole D/R |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.747/0.701 | +0.046 | 0.762 | 0.989 | 0.525/0.553 | 0.494/0.484 |
| 1 | 0.725/0.438 | +0.287 | 0.518 | 0.990 | 0.512/0.525 | 0.480/0.520 |
| 2 | 0.742/0.576 | +0.165 | 0.776 | 0.958 | 0.511/0.531 | 0.473/0.529 |
| 3 | 0.705/0.631 | +0.073 | 0.823 | 0.963 | 0.499/0.496 | 0.479/0.518 |
| 4 | 0.690/0.467 | +0.223 | 0.632 | 0.995 | 0.477/0.467 | 0.500/0.480 |
| 5 | 0.752/0.594 | +0.158 | 0.726 | 0.997 | 0.577/0.571 | 0.469/0.523 |
| 6 | 0.667/0.610 | +0.057 | 0.696 | 0.958 | 0.483/0.459 | 0.441/0.490 |
| 7 | 0.788/0.721 | +0.067 | 0.932 | 0.992 | 0.515/0.526 | 0.441/0.486 |
| 8 | 0.704/0.691 | +0.013 | 0.820 | 0.848 | 0.426/0.430 | 0.488/0.480 |
| 9 | 0.689/0.676 | +0.013 | 0.825 | 0.703 | 0.515/0.502 | 0.475/0.498 |
| 10 | 0.794/0.736 | +0.058 | 0.784 | 0.616 | 0.510/0.527 | 0.514/0.506 |
| 11 | 0.708/0.674 | +0.034 | 0.768 | 0.872 | 0.487/0.488 | 0.527/0.506 |
| 12 | 0.607/0.612 | -0.005 | 0.916 | 0.925 | 0.551/0.552 | 0.479/0.449 |
| 13 | 0.659/0.586 | +0.073 | 0.799 | 0.976 | 0.509/0.523 | 0.525/0.518 |
| 14 | 0.761/0.694 | +0.067 | 0.801 | 0.582 | 0.510/0.490 | 0.498/0.510 |
| 15 | 0.716/0.669 | +0.047 | 0.797 | 0.917 | 0.551/0.557 | 0.564/0.521 |
| 16 | 0.753/0.751 | +0.002 | 0.783 | 0.725 | 0.527/0.516 | 0.547/0.494 |
| 17 | 0.763/0.721 | +0.042 | 0.729 | 0.566 | 0.446/0.449 | 0.533/0.541 |

## Guardrail

A recurring complement is still a coordinate-level object, not a semantic feature. It earns fresh token decoding only after beating the generic-basis and destroyed-binding explanations. Because edges were selected for large residual first-order CE effect, effect magnitude itself is not an independent validation target.
