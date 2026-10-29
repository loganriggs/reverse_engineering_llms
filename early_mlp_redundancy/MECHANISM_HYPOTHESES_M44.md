# M44: refreshed mechanism hypotheses after native half repair

This is a working hypothesis set, not posterior probabilities. M41–M43 close the fixed half-scaling, prediction and interpolation sequence. The original broad goal remains: explain native computation and build faithful reusable representations. Do not equate an improved surrogate with that explanation.

## Twenty distinct hypotheses and discriminators

| # | Hypothesis | Current evidence / limitation | Discriminator |
|---|---|---|---|
| 1 | Near-identical channel replicas support averaging | MLP0 is compatible with fixed half compensation; MLP1/2 fail that sufficient signature | Other fixed partitions or direct function matching; do not infer identity from one mask |
| 2 | Shared-plus-private contributions explain distributed writes | Constructive energies, partial predictability, residual errors support a mixture | Separate shared signal from conditional residuals and test causal replacement |
| 3 | Token identity is a common cause of half correlation | Early writes strongly token-dependent; current regressions have no token control | Compare training-only token means and token-centered retained predictors |
| 4 | Contextual computation is shared beyond token identity | Natural context variation is substantial, but does not prove half sharing | Cross-document within-token predictive gains and native consequences |
| 5 | Rare/contextual directions are complementary rather than redundant | Full replacements fail despite average local gains | Family/input tails and conditional residual effects, all cases retained |
| 6 | The useful shared signal is low rank | Rank32 prediction improves local error | Test its incremental value beyond token means before any rank sweep |
| 7 | Reusable units overlap but need not form disjoint blocks | Learned overlaps beat random groups but lose scalar fidelity | Native shared/exclusive interventions and task-combination transfer |
| 8 | Stable subspaces admit unstable internal coordinates | Prior seed stability stronger for full spaces than individual blocks | Match subspaces/functions without selecting on evaluation signatures |
| 9 | Readout anisotropy breaks Euclidean reconstruction as a target | Local improvements coexist with harmful stable output slopes | Compare matched local-error orientations or native sensitivity metrics |
| 10 | Useful directions have incorrect causal magnitude | M43 gives widespread initial help followed by full harm | Supported within fixed assay; no further evaluation-amplitude tuning |
| 11 | Nonlinear downstream changes cause full-repair failures | Compatible, but quadratic readers already permit overshoot | Compare predicted local quadratic effects with finite native effects; not yet identified |
| 12 | Downstream normalization weakens amplitude loss but preserves directional harm | Earlier norm controls cause less damage | Match norms and orientations; normalization-specific mediation needed |
| 13 | Downstream modules actively compensate for deletion | Low output impact can be consistent with compensation | Freeze selected downstream responses versus live paths; tolerance alone insufficient |
| 14 | MLP2 contains locally cancelling subcircuits within net constructive totals | Aggregate energy does not rule out pairwise cancellation | Input-conditioned signed groups with held-out validation |
| 15 | Cancellation is the dominant early-layer mechanism | Weakened by matched net constructive energies and repair signatures | Require destructive aggregate or targeted subfunction evidence; not a default label |
| 16 | Coactivation groups mainly reflect frequency/norm rather than reusable functions | Coactivation improves prediction but does not validate native units | Frequency/token controls and norm-matched native intervention tests |
| 17 | Context determines how downstream readers use the same feature | Earlier identical native interventions change direction across contexts | Factorial context/task transfer with fixed interventions, not re-labeling by outcome |
| 18 | Binary-task margins hide distributional changes | Directly established by deletion tests | Keep full KL and local/causal errors beside accuracy |
| 19 | Spare capacity and robustness pressures favor spreading versus specialization | Controlled toys support this causal possibility | Controlled training with matched target complexity; not inferred for native training history |
| 20 | The current results are dominated by reused data, masks, or surrogate constraints | Real limitation: two dependent masks, familiar tasks, scalar-trained dictionary, fixed rank | Prespecified broader distributions/partitions after closing current controls; preserve whole-task holdouts |

## Most supported current account

Hypotheses2,9,10,17 and18 have convergent evidence at their stated scope: distributed partially aligned contributions, partial statistical predictability, context-sensitive downstream use, and a gap between geometric and behavioral fidelity. This is not a claim that semantic units have been recovered. Hypotheses3 and4 are the most informative unresolved fork: shared token identity could explain much of the apparent shared structure, or retained writes could convey useful context beyond it. Hypotheses11,13,14 and19 remain plausible but are not established native causal explanations.

## Next experiment: conditional sharing, not another fitted repair

Use only training-document token means and the six already frozen M42 maps. Compare the existing rank32 predictor with a missing-half token mean and a token-centered predictor:

    predicted missing = mean_missing(token) + (retained - mean_retained(token)) U B.

This uses the existing slope UB; there is no new ridge/rank fit. For unseen tokens use global training means, making the token-centered predictor exactly the original rank32 formula. Report coverage, known/unknown-token local errors, families, input tails and output KL separately. Native output interventions are essential: extra local predictability still need not be useful computation. All natural and task evaluation inputs remain familiar development data; existing task exclusions remain untouched.

A gain from token means alone supports token identity as a major common cause. A further token-centered gain supports conditional predictability beyond that baseline. Neither proves a semantic feature, native duplicate subfunction, or independence from context confounds. The deliverable to featurizers is a token-conditioned control for apparent shared directions, rather than adding another architecture based on raw coactivation.

Stop after this fixed comparison and its audit. Do not tune token thresholds, fit another rank, select favorable families, or optimize repair strength on the evaluation set. Shared/exclusive gate interventions remain pending; their interpretation should use this conditional baseline.

## Control construction completed

The frozen protocol and all six token-control states are built. Independent NumPy grouping reproduces every training token mean exactly; all original M42 map tensors remain bitwise unchanged. The4096training positions contain1760token types,1250of which appear once. Training token identity covers60.25% of natural evaluation positions. This sparsity makes training variance explained an optimistic description, not a validated token-table explanation. Evaluation must report coverage and fallback behavior explicitly; do not interpret uncovered tokens as failed context sharing.

Native transfer has not yet run. Next implement the three-arm evaluator and independent audit, require exact replay of original M42 rank32 rows, and retain the whole familiar task panel. The token controls use additional storage and are diagnostic, not a parameter-matched architecture comparison.

[Protocol](validation/native_half_token_protocol.json) · [Control states and coverage](validation/native_half_token_controls.json) · [Construction audit](validation/native_half_token_controls_audit.json)
