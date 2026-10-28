# Split-stability red team of the tangent-bundle quotient

Four rank-64 basis triples are fit to different 192-of-256 discovery-site subsets. Each is tested on its complementary 64 sites and on the same disjoint external corpora. The swapped-role quotient has exactly the same ranks and learned subspaces as the positive condition.

## Locked hypotheses

- **R0_all_split_bases_are_discovery_only: supported.** Every split fits exactly 192 of 256 discovery sites and evaluates its complementary 64 sites without overlap.
- **R1_score_fidelity_is_split_stable: supported.** Across split choices, the external-corpus range of prediction and target fidelity is at most 0.05 in every corpus.
- **R2_margin_fidelity_is_split_stable: supported.** Across split choices, the external-corpus range of margin fidelity is at most 0.10 in every corpus.
- **R3_every_split_recovers_scores: supported.** Every split's rank-64 role quotient retains at least 0.80 prediction and target fidelity on every external corpus.
- **R4_correct_roles_beat_swapped_roles: supported.** For every split and external corpus, correct assignment exceeds swapped assignment by at least 0.10 in mean endpoint fidelity.
- **R5_role_bases_are_subspace_stable: not supported.** Relative to the head split, every context, tangent, and output rank-64 basis has normalized projector overlap at least 0.75.
- **R6_margin_is_systematically_harder: not supported.** In every split and external corpus, margin fidelity is at least 0.15 below the mean of prediction and target fidelity.
- **R7_heldout_split_predicts_external_behavior: supported.** Across split choices, mean validation fidelity and mean held-out FineWeb fidelity have positive Spearman correlation.

## External role-separated fidelity

Each cell is prediction/target/margin.

| Split | FineWeb | WikiText | TinyStories | Validation |
|---|---:|---:|---:|---:|
| head | 0.899/0.886/0.603 | 0.941/0.930/0.604 | 0.955/0.951/0.809 | 0.890/0.846/0.295 |
| tail | 0.891/0.880/0.604 | 0.938/0.927/0.599 | 0.962/0.959/0.821 | 0.831/0.809/0.262 |
| interleaved | 0.898/0.887/0.591 | 0.927/0.921/0.569 | 0.954/0.952/0.810 | 0.878/0.838/0.422 |
| random | 0.904/0.892/0.593 | 0.938/0.929/0.590 | 0.958/0.955/0.827 | 0.845/0.823/0.312 |

## Basis stability and matched role control

| Split | context/tangent/output overlap with head | Mean role − swapped: FW/Wiki/Tiny |
|---|---:|---:|
| head | 1.000/1.000/1.000 | 0.924/0.851/0.599 |
| tail | 0.632/0.622/0.667 | 0.892/0.807/0.620 |
| interleaved | 0.720/0.744/0.824 | 0.924/0.814/0.639 |
| random | 0.722/0.702/0.770 | 0.940/0.845/0.570 |

Validation-to-external-FineWeb Spearman across four splits: `0.200`.

Stable external behavior with imperfect basis overlap would mean that the functional quotient is more identifiable than its individual PCA axes. An unstable margin with stable score paths would reinforce precision-conditioned support rather than refute the score-level quotient.
