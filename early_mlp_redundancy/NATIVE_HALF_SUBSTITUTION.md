# M42: predictable missing contributions do not guarantee functional substitution

M41 showed that simple doubling of surviving channels helps MLP0 but harms MLP1. This fixed follow-up tests whether a learned low-complexity transformation can recover information that identity scaling misses.

## Mechanistic hypothesis and implementation

Let X be the retained half’s native product write and Y the missing half’s write. A conditional predictor of Y from X can outperform the unconditional mean even when Y is not approximately X. This is a broader form of statistical redundancy than literal duplication. It still need not preserve the output distribution: local Euclidean error and downstream importance weight directions differently.

For each MLP0/1/2 and each of the two complementary half masks, fit the top32 PCA directions U of centered X on4096 positions from natural documents0–31. Fit ridge regression from Z=(X-mean_X)U to centered Y, using lambda=.001 trace(Z^T Z)/32. All ranks, masks and ridge choices were frozen before fitting; all six maps retained. The result is mean_Y+(X-mean_X)UB. No task data or evaluation documents entered the fit. The two matrix factors and means contain76,032 stored scalars per map (some are constrained/derived, not independent trainable parameters).

Compare mean-only and rank32 replacement of the missing half, retaining the surviving native write and output bias. Prediction uses only retained write and training constants; native missing writes are used for scoring. All downstream computation stays live. Compare with M41 raw deletion and doubling on identical inputs. This is learned repair with additional computation, not literal pruning robustness.

## Results

Local error is pooled squared product-write error / native product-write energy. KL is mean native||edited full-output KL over positions, inputs, and the two complementary masks. Task train/evaluation names refer to the pre-existing task-source split; neither task partition trained these maps. All data remain familiar development material.

| Cohort | Layer | Raw deletion KL | Mean repair KL | Rank32 repair KL | Mean local error | Rank32 local error |
|---|---|---:|---:|---:|---:|---:|
| task train | MLP0 | 0.04038 | 0.02670 | 0.02568 | 0.1854 | 0.1036 |
| task train | MLP1 | 0.03689 | 0.05269 | 0.05247 | 0.3185 | 0.2159 |
| task train | MLP2 | 0.02303 | 0.02705 | 0.02380 | 0.3538 | 0.2947 |
| task evaluation | MLP0 | 0.03215 | 0.02877 | 0.02643 | 0.1815 | 0.1040 |
| task evaluation | MLP1 | 0.03738 | 0.05267 | 0.06308 | 0.3161 | 0.2218 |
| task evaluation | MLP2 | 0.02350 | 0.02511 | 0.02788 | 0.3490 | 0.2925 |
| natural evaluation | MLP0 | 0.02885 | 0.03853 | 0.02866 | 0.1784 | 0.1097 |
| natural evaluation | MLP1 | 0.03282 | 0.08331 | 0.06474 | 0.3168 | 0.2268 |
| natural evaluation | MLP2 | 0.02264 | 0.03842 | 0.03338 | 0.3450 | 0.2917 |

Rank32 prediction reduces local error versus mean-only repair in all nine layer/cohort aggregates. On natural evaluation data it improves output KL versus mean-only repair for all three layers, but versus raw deletion it is roughly tied for MLP0 and worse for MLP1/2. MLP1 natural KL rises from .03282 under deletion to .06474 under learned repair. On source-separated task evaluation, MLP1/2 learned repair is worse than both mean-only repair and raw deletion.

Mask direction matters. On task evaluation, MLP0 learned/raw KL ratios are1.234 and.476 for the two masks: the pooled improvement is not uniform. MLP1 learned/raw ratios exceed1 for both masks in all cohorts; MLP2 does so for both masks on natural and task evaluation data. These are descriptive results for one fixed complementary partition, not uncertainty estimates or statements about every subset.

## Conceptual update

The surviving and missing halves share some predictable variation, but predictable variation is not sufficient for functional substitutability. The evidence supports distinguishing at least: (1) distributed implementation; (2) simple averaging redundancy; (3) conditional statistical predictability; and (4) successful causal replacement under a stated fidelity criterion. MLP1 shows the third more clearly than the second or fourth in these tests.

A theoretical explanation compatible with the result is anisotropic downstream sensitivity: reducing Euclidean error can increase error along directions that readers care about. Finite replacement also permits nonlinear effects. This experiment does not isolate those explanations, establish cancellation, or prove that no better replacement exists. PCA directions optimized for input variance need not be the best predictive or causally faithful directions. No architecture-wide conclusion follows from one rank and ridge choice.

## Audit and stopping criterion

Training used32 native forwards; transfer used2,496; training audit used one additional native forward. All six PCA eigenspectra and coefficient matrices were independently verified using NumPy and augmented least squares (maximum coefficient relative discrepancy4.04e-15). Six first-document native half captures replay exactly. Document separation and training-only provenance checked.

Transfer audit checked all2,304rows, inventories, summaries, native source energies and native task outputs against M41. Twelve samples independently reproduce prediction exactly and output metrics within1.19e-8. This is not an independent transformer implementation or validation of arbitrary intervention-response fidelity.

Stop rank/ridge optimization on this panel. Next discriminate whether the harmful repair direction is already harmful at small amplitude or only after a large move, using a fixed interpolation from raw deletion toward these frozen predictions, native outputs as reference, and all six maps retained. This can distinguish immediate directional mismatch from finite-amplitude overshoot without fitting another representation. Shared/exclusive featurizer interventions and whole-task discovery remain open.

[Protocol](validation/native_half_substitution_protocol.json) · [Training](validation/native_half_substitution_training.json) · [Training audit](validation/native_half_substitution_training_audit.json) · [Transfer](validation/native_half_substitution_transfer.json) · [Transfer audit](validation/native_half_substitution_transfer_audit.json)
