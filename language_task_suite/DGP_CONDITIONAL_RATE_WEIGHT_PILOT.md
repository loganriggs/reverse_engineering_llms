# DGP rule prevalence and weight recovery: V14 pilot

This is a controlled first step in the [DGP-to-weight protocol](DGP_TO_WEIGHT_INTERPRETABILITY_PROTOCOL.md), **not** the more realistic language generator. The V14 sampled stream is still repetitive and far from FineWeb. We changed one data-generating-process parameter: after an item-attribute fact, the probability of continuing with `so PERSON ACTION it` was 0.10 or 0.60. The action is deterministically selected from the attribute (for example, `broken → repairs`). Both streams retain V14's other jointly sampled tasks. Architecture, initial seed, optimizer, 600 updates, 256-token contexts, and training budget were matched. This compares exposure to the same rule, not the universal learnability of that rule.

The [generator audits](results/v14_dgp_conditional_rate010_audit_n64.json) at 0.10 and [0.60](results/v14_dgp_conditional_rate060_audit_n64.json) replay sampled probabilities and supported edits. Across two disjoint 64-document panels, the low rate produced 6 and 9 action sites; the high rate produced 47 and 67. Each reported local edit had oracle action-probability response 1. This confirms that the intervention substantially changed rule prevalence while preserving the rule and its causal test. The altered clause length and word frequencies also change somewhat; this is not a perfectly isolated intervention on training-sample count.

## Behavioral eligibility before weight analysis

Two depth-2, width-64, four-head bilinear models were trained from the same model seed using full next-token cross-entropy. Both use RMSNorm. The [low-rate training record](results/v14_dgp_conditional_rate010_depth2_seed3_600/results.json) and [high-rate record](results/v14_dgp_conditional_rate060_depth2_seed3_600/results.json) contain evaluation traces and checkpoints.

| 512-document fresh reference panel | Low-rate dense, 0.10 | High-rate dense, 0.60 | High-rate, 60%-zero attention + no-L1 recovery |
| --- | ---: | ---: | ---: |
| Seed 99941 action sites | 79 | 455 | 455 |
| Action-site cross-entropy (nats) | 2.213 | 0.094 | 0.205 |
| Supported action edits | 79 | 128 sampled | 128 sampled |
| Mean rise in newly correct action probability | 0.015 | 0.895 | 0.772 |
| Seed 99942 action sites | 65 | 423 | 423 |
| Action-site cross-entropy (nats) | 1.984 | 0.104 | 0.191 |
| Mean rise in newly correct action probability | 0.007 | 0.896 | 0.790 |

The two [low-rate](results/v14_dgp_conditional_rate010_depth2_seed3_600/fresh_panel_99941.json), [high-rate](results/v14_dgp_conditional_rate060_depth2_seed3_600/fresh_panel_99941.json), and [sparse](results/v14_dgp_conditional_rate060_sparse_seed99441/fresh_panel_99941.json) panel records include the other task families and diagnostic-regime scores. The high-rate model's earlier-fact adjective/object causal responses remain near zero, so those rules are **not** eligible for weight-circuit claims. The low-rate model did not learn the local action rule at this budget; inspecting its weights for that rule would be unjustified. These are two evaluation-panel seeds but only **one training seed** per DGP setting.

On a common, intermediate 0.35-rate reference panel, the [low-trained](results/v14_dgp_conditional_rate010_depth2_seed3_600/fresh_common_rate035_panel_99951.json) and [high-trained](results/v14_dgp_conditional_rate060_depth2_seed3_600/fresh_common_rate035_panel_99951.json) models saw the same 258 action sites. Their action cross-entropies were 2.030 and 0.102 nats, and their mean supported-edit responses were 0.015 and 0.888. Thus the difference is not just an artifact of evaluating each model at its own training rate.

The common-panel run found four supported earlier-fact edits with zero effect on the chosen adjective target: changing a fact also changed whether its item had already been mentioned. The evaluator now excludes and counts such no-op pairs; causal scores are computed only on rule contrasts with an actual oracle target effect. This correction does not alter the local-action results.

## Sparse-weight result and limitation

The [sparsification record](results/v14_dgp_conditional_rate060_sparse_seed99441/results.json) used attention-projection L1 training, 40% and 60% global attention-weight pruning, then 200 updates with **no L1** and frozen masks. Embedding, unembedding, and normalization parameters remained dense. The 60%-zero stage initially raised action cross-entropy to 3.042 nats on its fixed evaluation stream; no-L1 recovery brought it to 0.200. On the two untouched causal panels above, the recovered sparse model retains much but not all of the dense response. The masks leave 19,661/49,152 attention weights and 69,261/98,752 total parameters live.

The [action-map and edge-ablation figure](results/v14_dgp_conditional_rate060_sparse_seed99441/v14_sparse_action_map_and_edges.png) compares planted mapping, dense/sparse full-model probabilities, and removal of the attribute-token attention edge in each head. A query self-edge removal is the control. On six attributes × 16 fresh examples, the average absolute target-probability drop from removing the attribute edge is concentrated in second-layer head 3: 0.763 in the dense model and 0.710 in the sparse model. The other seven heads are at most 0.011 in the sparse model; sparse self-edge controls are at most 0.010. This supports a causal path for the **learned local rule**, but the dense model already localized it to the same head. The present pruning result establishes a sparse, partly faithful representation; it does not demonstrate improved modularity, circuit uniqueness, or recovery of the unlearned cross-sentence rules.

Next, replicate the prevalence effect across training seeds and training-token budgets, then compare weight-only canonical/bond bases and task/data-weighted bases at matched functional fidelity on the **same high-rate learned rule**. In parallel, the main DGP work must vary whole-sentence frames, discourse, lexical burstiness, and cross-rule coupling; this V14 parameter sweep alone does not repair the realism gap.
