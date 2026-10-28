## H727 — Frozen overlapping groups preserve some computation, but scalar supports remain stronger

We tested every H726 grouping without refitting: all three scalar-dictionary seeds, raw and token/position-conditioned memberships, three degree-preserving rewires of each, and random disjoint partitions. We intervened at two previously verified attention heads: layer 11 head 3 for agreement and layer 13 head 0 for capital recall. Identity and scalar Top-8 controls were rerun.

For task transfer, F is the frozen featurizer reconstruction. We add F(donor head) minus F(receiver head) to the native receiver head, then allow the downstream model to run. This tests which counterfactual information the reconstruction preserves; it does not directly test individual shared atoms or establish a computational DAG. Separately, replacing the native head by F(native head) on 32 natural contexts measures output KL and observed next-token cross-entropy change.

Results below average all dictionary seeds and, for rewires, all three controls per seed. Task fraction is the mean donor-directed margin change divided by the exact-head intervention's mean change on eight transfer pairs per family. It is not task accuracy. KL is native-to-reconstructed output-distribution KL; lower is better.

| Featurizer | Agreement task fraction | Capital task fraction | Agreement KL | Capital KL |
|---|---:|---:|---:|---:|
| Scalar Top-8 | 94.0% | 67.1% | 0.000526 | 0.001949 |
| Raw overlapping groups | 90.1% | 53.8% | 0.000812 | 0.002155 |
| Raw degree-matched rewires | 88.4% | 49.0% | 0.000764 | 0.002649 |
| Conditioned overlapping groups | 92.9% | 54.5% | 0.000820 | 0.002543 |
| Conditioned degree-matched rewires | 89.5% | 49.4% | 0.000810 | 0.002633 |
| Random disjoint groups | 90.4% | 47.1% | 0.000948 | 0.002915 |

Raw and conditioned groups preserve modestly more task effect than their rewired controls. Their mean task-minus-absolute-nuisance specificity is 4.314/4.453 for agreement and 1.704/1.727 for capitals; scalar Top-8 achieves 4.502 and 2.102. Across individual transfer interventions, relative RMS effect error is 0.110/0.103 agreement and 0.476/0.465 capitals for raw/conditioned groups, versus scalar 0.079/0.342. No inferred-group transfer effect reverses the corresponding full-head effect's sign.

The functional advantage over rewires does not uniformly translate to output fidelity: agreement KL is slightly worse for inferred groups. Mean observed next-token CE change is +0.00538/+0.00484 agreement and +0.00865/+0.01150 capitals for raw/conditioned groups; scalar changes are +0.00194/+0.00707. These small-panel means are descriptive, not independently confirmed population improvements.

Implementation checks: 4,992 full-model example forwards completed; all source/input hashes remained unchanged and KL values were finite. Exact-head task replay maximum error was 3.81e-6, identity reconstruction maximum KL 3.64e-12, and scalar task results reproduced H725 exactly. H726 union/gating unit tests also passed. No new documents or reserved-test examples were used.

Interpretation: there is useful support structure beyond membership-degree statistics, especially for capital recall, but these overlapping groups are not better featurizers overall than unrestricted scalar Top-8. They are not yet verified manifolds. This does not rule out overlapping structure: greedy coordinate-energy gates may be poorly matched to a nonorthogonal decoder, and heuristic memberships may be suboptimal.

Next informative experiment: distinguish selection error from membership limitations using reconstruction-aware selection among the exact same frozen groups, with identical active-group and membership budgets. Compare to degree-matched rewires and unrestricted scalar supports; use training for any optimization and validation for evaluation. Then take only reproducible, faithful candidates into selective shared-component interventions and fresh diverse-task tests, separating linear subspaces, curved manifolds, taxonomy gates, and actual reusable computations. The present two-head panel is a targeted assay and does not meet the broader diverse-task validation requirement by itself.
