# A rule-aware prior rescues a severely reduced OV circuit

The prior [OV reduction experiment](THREE_RULE_OV_REDUCTION_RESULT.md) found that unsupervised activation-weighted reconstruction can discard a learned bracket rule. This follow-up fixes the rank at **two value channels per head** in both attention layers, starts from the ordinary weight-SVD truncation, and updates only those retained OV factors. All QK, MLP, embedding, unembedding, and discarded value coordinates remain frozen. The discarded-coordinate maximum is exactly zero after training in both runs.

The fit objective is teacher-to-student KL at bracket-closer positions on 80 train-partition reports with all eight alias×bracket×capitalization arms. It uses 300 Adam steps. A separate 40-report heldout partition tests paired bracket edits and teacher KL at bracket, alias, capitalization, and sampled background positions. The two checkpoints use the same panel-generating seed; their native behaviors differ. This is a task-supervised compression test, not unsupervised rule discovery.

| Checkpoint | Native bracket paired exact | Rank-2 initial | Rank-2 after rule KL fit | Bracket KL initial → final | Background KL initial → final |
|---|---:|---:|---:|---:|---:|
| 55301 | .931 | .700 | .950 | .511 → .032 | .346 → .379 |
| 55302 | .906 | .625 | .913 | .963 → .172 | .850 → 1.154 |

The fit recovers heldout bracket behavior at a severe rank constraint, with a clear cost to ordinary language predictions, especially for checkpoint 55302. The final paired scores slightly exceed native scores on this small panel; that is finite-panel variation, not evidence that compression improves the original model. Alias remains unsolved in the teacher, and the capitalization site has a common-word shortcut, so neither supports a circuit recovery claim here.

Together with the rank-4 comparison, this distinguishes three objectives: (1) weight reconstruction, (2) average data reconstruction, and (3) preservation of a specified behavioral rule. The third works for a known learned rule but requires labels/sites supplied by the DGP. To test interpretability rather than only task-specific distillation, the next experiment should compare whether a low-rank or sparse component discovered from one rule transfers to held-out combinations or another independently learned rule. It should also include a background-KL constraint or Pareto curve, since site-only fitting can damage the rest of the language model.

Reproduce with `fit_three_rule_bracket_ov.py`; aggregate outputs are `results/fit_three_rule_bracket_ov_seed55301_rank2_n80_n40.json` and the corresponding seed55302 file. The corpus-derived report text and model checkpoints remain local.
