# Weight-only prediction of CE-selected route subspaces

Each head is evaluated as a symmetry-invariant random function of its two QK factors and OV map. The leading eigenvectors of the nine-head operator Gram matrix define a static weight-only basis in the same coordinate space as the CE-selected routes. Two independent sketches test Monte Carlo stability.

## Locked hypotheses

- **W0_weight_bases_are_sketch_stable: supported.** Observed: `{'layers': [0, 1, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]}`
- **W1_weight_geometry_predicts_ce_subspaces_above_random: not supported.** Observed: `{'layers': [3, 10, 12, 14, 15]}`
- **W2_scale_information_improves_weight_prediction: not supported.** Observed: `{'layers': [0, 1, 3, 9, 10, 11, 14, 15, 17]}`
- **W3_weight_basis_recovers_material_ce_residual_geometry: supported.** Observed: `{'layers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}`
- **W4_early_ce_novelty_is_visible_in_weights: not supported.** Observed: `{'layers': []}`
- **W5_qk_and_ov_roles_are_distinguishable: not supported.** Observed: `{'layers': [0, 1, 2, 3, 5, 11, 12]}`

| L | CE/generic fidelity | QK cos/fid | OV cos/fid | Combined cos/fid | Scaled cos/fid/stability | Best vs random p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.747/0.701 | 0.574/0.547 | 0.753/0.576 | 0.657/0.579 | 0.686/0.566/0.970 | 0.753/0.761 |
| 1 | 0.725/0.438 | 0.535/0.496 | 0.686/0.560 | 0.629/0.564 | 0.679/0.572/0.990 | 0.686/0.764 |
| 2 | 0.742/0.576 | 0.550/0.418 | 0.721/0.579 | 0.578/0.454 | 0.544/0.370/0.766 | 0.721/0.752 |
| 3 | 0.705/0.631 | 0.618/0.491 | 0.783/0.587 | 0.749/0.556 | 0.800/0.558/0.990 | 0.800/0.758 |
| 4 | 0.690/0.467 | 0.735/0.584 | 0.691/0.556 | 0.679/0.577 | 0.583/0.428/0.967 | 0.735/0.753 |
| 5 | 0.752/0.594 | 0.584/0.535 | 0.697/0.598 | 0.683/0.602 | 0.456/0.345/0.997 | 0.697/0.751 |
| 6 | 0.667/0.610 | 0.619/0.484 | 0.673/0.514 | 0.736/0.528 | 0.551/0.381/0.869 | 0.736/0.759 |
| 7 | 0.788/0.721 | 0.635/0.535 | 0.635/0.483 | 0.636/0.561 | 0.483/0.565/0.998 | 0.636/0.762 |
| 8 | 0.704/0.691 | 0.727/0.503 | 0.652/0.415 | 0.750/0.519 | 0.575/0.460/0.940 | 0.750/0.764 |
| 9 | 0.689/0.676 | 0.639/0.519 | 0.657/0.579 | 0.658/0.561 | 0.671/0.581/0.886 | 0.671/0.761 |
| 10 | 0.794/0.736 | 0.678/0.573 | 0.757/0.673 | 0.734/0.672 | 0.803/0.608/0.989 | 0.803/0.757 |
| 11 | 0.708/0.674 | 0.507/0.376 | 0.648/0.506 | 0.621/0.481 | 0.694/0.574/0.993 | 0.694/0.756 |
| 12 | 0.607/0.612 | 0.541/0.412 | 0.745/0.518 | 0.756/0.521 | 0.648/0.522/0.996 | 0.756/0.749 |
| 13 | 0.659/0.586 | 0.717/0.493 | 0.638/0.520 | 0.754/0.552 | 0.475/0.373/0.945 | 0.754/0.762 |
| 14 | 0.761/0.694 | 0.658/0.681 | 0.613/0.642 | 0.737/0.678 | 0.774/0.672/0.938 | 0.774/0.753 |
| 15 | 0.716/0.669 | 0.560/0.497 | 0.600/0.489 | 0.558/0.505 | 0.942/0.683/0.992 | 0.942/0.750 |
| 16 | 0.753/0.751 | 0.716/0.598 | 0.690/0.586 | 0.698/0.615 | 0.553/0.625/0.998 | 0.716/0.749 |
| 17 | 0.763/0.721 | 0.512/0.508 | 0.609/0.619 | 0.654/0.617 | 0.709/0.648/0.986 | 0.709/0.750 |

## Guardrails

The random-function basis is fully weight-only and symmetry invariant, but its isotropic input distribution need not match FineWeb states or the next-token loss. High canonical cosine would show that static operator geometry constrains the executable CE basis; low cosine would not make the CE basis uninterpretable, but would establish that context/frequency/loss weighting selects a different coordinate system. Choosing the best of four weight views receives an explicit layer-matched random-subspace comparison.
