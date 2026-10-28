

## H282 — These alternative feature families do not outperform the existing reads

H281 motivated a feature-family discriminator. Three adapters share the same frozen H279 base,768 training/256 validation documents,64 features and73,728 learned output coefficients. Existing linear H280 fits are reused. The quadratic family standardizes the existing64 reads, forms2080 upper-triangle products, centers them and takes64 leading training-covariance PCs. The outside family projects normalized inputs orthogonally off the existing read span and takes64 leading training-covariance PCs. Both maps are centered/standardized using training states only. These are unsupervised variance-selected maps, not optimal response-selected features. Quadratic features may correlate with linear features on the natural distribution; this is not an orthogonal decomposition of all model error. Outside reads exclude the original span to floating-point precision.

Each new family uses five ridge fits and eight KL runs,400 updates each, identical seed/batches and validation selection to H280. Output coefficient counts match; feature-map construction, stored map size and arithmetic do not. Maps currently store explicit projection matrices, not storage-minimized exports. No new combined-family model was fit. Models are frozen before fresh evaluation. Three directional contrast definitions and practical5% thresholds were fixed in configs;97.5% per-contrast bootstrap intervals are not simultaneous familywise intervals. Repair requires80% KL recovery, CE<=.02 and top1>=.98. The family for repair is chosen by validation, never fresh results.

Fresh128 documents exclude2624 earlier unique complete-document hashes. Independent full-document/window/file hash and window-content audits passed. This is not a model-training holdout. A meaningful synthetic test verifies maps depend only on training rows, saved maps reproduce features, within-span quadratic features ignore orthogonal perturbations, and outside features ignore within-span perturbations. Its initial80-dimensional fixture left only16 outside dimensions for64 PCs and failed; changing the fixture to160 dimensions resolved the invalid rank assumption. Production input dimension1152 leaves1088 outside dimensions. A shell launch initially failed because unqualified python was absent; the workspace venv was then used. No production fit was restarted for these fixture/launch issues.

| Family | KL train | KL validation | Fresh MSE KL | Fresh KL-adapter KL | Fresh CE increase | Top1 agreement |
|---|---:|---:|---:|---:|---:|---:|
|linear|0.034373|0.052036|0.057158|0.055234|0.03732|89.84%|
|quadratic|0.039468|0.053218|0.059342|0.057739|0.03598|89.45%|
|outside|0.032173|0.052694|0.059409|0.059577|0.04218|88.67%|

- quadratic_gain: -4.54% relative KL reduction; paired97.5% [-9.84,0.74]%.
- outside_gain: -7.86% relative KL reduction; paired97.5% [-15.07,-1.34]%.
- family_difference: 3.08% relative KL reduction; paired97.5% [-2.60,8.33]%.
- linear_shuffle: 16.68% relative KL reduction; paired97.5% [11.00,22.07]%.
- quadratic_shuffle: 14.00% relative KL reduction; paired97.5% [8.54,19.80]%.
- outside_shuffle: 9.97% relative KL reduction; paired97.5% [4.29,15.24]%.

Selections: {"linear": {"mse": "mse_a1.0", "kl": "mse_lr0.3_p0.1", "checkpoint": 300}, "quadratic": {"mse": "mse_a1.0", "kl": "mse_lr0.3_p0", "checkpoint": 400}, "outside": {"mse": "mse_a0.1", "kl": "mse_lr0.3_p0.1", "checkpoint": 400}}. Validation-selected family: linear. All four gates FAIL. No thresholds relaxed. Shuffle diagnostics cyclically mismatch document features while preserving position slots; they are not semantic interventions.

**Inference.** Neither tested alternative beats the existing linear adapter. The outside family lowers training KL most, yet transfers worse, reinforcing an estimation/generalization limitation for this map. This weakens the claim that simply adding variance-selected outside directions or quadratic PCs solves the remaining error at this budget. It does not rule out better feature selection, more data, complementary effects when combined, nonlinear readouts, or different regularization. Semantic-unit identification remains open. The comparison tests particular algorithms, not exhaustive representational classes.

**Next discriminator.** Before another architecture sweep, measure complementarity of the already frozen predictions. On the existing validation panel only, choose a convex mixture of the three predicted corrections (simplex grid step.1, plus the zero-correction base). Evaluate its frozen coefficients on a new panel; compare with the validation-selected single family. This adds only mixture coefficients and no new native teacher-fitting run. A gain would show that individually weaker families carry complementary information; no gain would narrow the utility of this particular family pool. Then measure source-response transfer before promoting a mixture. Use an oracle per-document selector only as a labeled diagnostic upper envelope, never a deployable result. Lock practical thresholds before fresh outcomes.

Original weights/compiler/base unchanged. All training and fresh evaluation jobs completed. Arrays, per-document metrics, frozen maps and selection curves retained. Goal remains active.
