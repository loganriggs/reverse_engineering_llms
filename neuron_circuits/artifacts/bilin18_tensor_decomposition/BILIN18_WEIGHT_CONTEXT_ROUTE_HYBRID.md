# Fixed-rank weight/context route hybrid

The route code always has rank four. `k` leading coordinates come from the discovery-selected static weight view; the remaining `4-k` coordinates are PCA directions fit to the discovery CE-residual routes after removing those weight coordinates. Every hybrid is frozen before replication-route evaluation.

## Locked hypotheses

- **H0_hybrid_bases_transfer: supported.** Observed: `{'fit': 'discovery', 'evaluation': 'replication'}`
- **H1_two_context_coordinates_are_sufficient: supported.** Observed: `{'layers': [0, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}`
- **H2_one_context_coordinate_is_not_generally_sufficient: supported.** Observed: `{'passing_layers': [11, 15]}`
- **H3_early_ce_novelty_requires_context_coordinates: supported.** Observed: `{'layers': [1, 2, 4, 5]}`
- **H4_context_correction_improves_static_weights: not supported.** Observed: `{'layers': [0, 3, 5, 7, 8, 10, 11, 16]}`

| L | Weight view | CE fidelity | Fidelity for weight coordinates k=0/1/2/3/4 | Cosine k=0/1/2/3/4 | Min context coords for 95% |
|---:|---|---:|---:|---:|---:|
| 0 | ov | 0.747 | 0.747/0.740/0.711/0.642/0.576 | 1.000/0.874/0.739/0.784/0.753 | 2 |
| 1 | ov | 0.725 | 0.725/0.676/0.651/0.577/0.560 | 1.000/0.856/0.785/0.618/0.686 | 4 |
| 2 | ov | 0.742 | 0.742/0.729/0.695/0.625/0.579 | 1.000/0.985/0.827/0.734/0.721 | 3 |
| 3 | scaled_combined | 0.705 | 0.705/0.701/0.694/0.614/0.558 | 1.000/0.941/0.936/0.800/0.800 | 2 |
| 4 | qk | 0.690 | 0.690/0.648/0.605/0.578/0.584 | 1.000/0.783/0.798/0.757/0.735 | 4 |
| 5 | ov | 0.752 | 0.752/0.747/0.699/0.707/0.598 | 1.000/0.996/0.869/0.767/0.697 | 3 |
| 6 | combined | 0.667 | 0.667/0.661/0.589/0.575/0.528 | 1.000/0.975/0.756/0.793/0.736 | 3 |
| 7 | combined | 0.788 | 0.788/0.746/0.753/0.684/0.561 | 1.000/0.780/0.741/0.658/0.636 | 2 |
| 8 | combined | 0.704 | 0.704/0.698/0.642/0.589/0.519 | 1.000/0.874/0.763/0.662/0.750 | 3 |
| 9 | scaled_combined | 0.689 | 0.689/0.693/0.660/0.627/0.581 | 1.000/0.938/0.778/0.744/0.671 | 2 |
| 10 | scaled_combined | 0.794 | 0.794/0.751/0.741/0.673/0.608 | 1.000/0.850/0.826/0.744/0.803 | 4 |
| 11 | scaled_combined | 0.708 | 0.708/0.712/0.694/0.691/0.574 | 1.000/0.934/0.851/0.844/0.694 | 1 |
| 12 | combined | 0.607 | 0.607/0.603/0.576/0.518/0.521 | 1.000/0.916/0.902/0.834/0.756 | 3 |
| 13 | combined | 0.659 | 0.659/0.644/0.608/0.581/0.552 | 1.000/0.966/0.857/0.803/0.754 | 3 |
| 14 | scaled_combined | 0.761 | 0.761/0.748/0.698/0.710/0.672 | 1.000/0.821/0.710/0.738/0.774 | 3 |
| 15 | scaled_combined | 0.716 | 0.716/0.693/0.694/0.702/0.683 | 1.000/0.967/0.969/0.964/0.942 | 0 |
| 16 | qk | 0.753 | 0.753/0.745/0.738/0.668/0.598 | 1.000/0.980/0.863/0.738/0.716 | 2 |
| 17 | scaled_combined | 0.763 | 0.763/0.749/0.745/0.692/0.648 | 1.000/0.844/0.758/0.787/0.709 | 2 |

## Interpretation

The context coordinates are fixed distribution-level corrections learned from FineWeb CE-important routes, not per-example dynamic directions. If a small correction transfers, the right ontology is a stable weight prior plus a corpus/loss-conditioned chart. If nearly all four coordinates must be refit, static operator geometry describes capability but not the executed basis.
