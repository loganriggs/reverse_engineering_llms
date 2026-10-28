## H898 — Coverage helps live distribution fidelity, but rank1 can preserve native behavior while distorting interventions

The previous turn made progress by isolating output coverage/rank at a fixed number axis in H897. This experiment runs all six frozen output fits at ranks1 and7 through the actual downstream model. The original A input axis, origin and number centroids stay fixed. Compare native cross-term replacement and opposite-number coordinate intervention with exact and cross-term-removal controls. Context-only and quadratic terms remain exact. Each intervention changes only the final-query L12 MLP output; downstream computation remains live.

Evaluation is the64-context H893 panel: eight noun pairs, two grammar templates G/H, matched copying, both number endpoints. Nouns and grammar templates are absent from these output fits, but the panel has been used in earlier experiments; this is diagnostic transfer, not globally untouched confirmation. No outcome exclusions or fit selection.

### Pooled grammar results

Native KL ratio divides replacement KL by removal KL. Effect RMSE compares the exact versus approximate reduction in expected-answer logit margin under number flipping, each relative to its own native or replaced baseline. Flip KL compares exact edited versus approximate edited distributions. The pooled panel includes failed-gate template H, so see the competent subset below.

|Output fit|Native KL ratio|Flip-effect RMSE|Exact-flip to approximate-flip KL|
|---|---:|---:|---:|
|exact|0.000|0.000|0.000000|
|drop_cross|1.000|1.156|0.003288|
|four_A_rank1|0.288|0.161|0.000674|
|four_A_rank7|0.254|0.144|0.000600|
|four_AB_rank1|0.295|0.836|0.001143|
|four_AB_rank7|0.153|0.161|0.000363|
|four_ABCDEF_rank1|0.176|0.226|0.000440|
|four_ABCDEF_rank7|0.133|0.104|0.000310|
|eight_A_rank1|0.271|0.142|0.000632|
|eight_A_rank7|0.233|0.127|0.000546|
|eight_AB_rank1|0.304|0.897|0.001248|
|eight_AB_rank7|0.149|0.116|0.000341|
|eight_ABCDEF_rank1|0.167|0.194|0.000407|
|eight_ABCDEF_rank7|0.132|0.108|0.000309|

With eight nouns, broadening A-only to A–F at rank7 reduces pooled native KL0.000324→0.000183 and exact-flip approximation KL0.000546→0.000309. Effect RMSE falls0.127→0.108. The local reconstruction improvement therefore partly survives downstream propagation, but not with equivalent magnitude on every metric.

The rank1 A/B fit exposes a major separation: eight-noun A/B native KL is0.000422, comfortably below removal0.001391, yet flip-effect RMSE is0.897 versus0.142 for eight-noun A rank1. Broader discovery at fixed rank can sacrifice a causally relevant direction even while remaining a relatively low-disruption native replacement. The explanation is not established by this observation; PCA ordering and downstream sensitivity are candidates.

### Competent grammar subset and copying

Grammar G retains100% binary is/are preference and56.25% exact-answer top1, passing the prior task gate. Grammar H retains100% binary preference but0% exact-answer top1, failing that gate. Copy G/H have87.5%/93.75% exact-answer top1. All rows remain included in the full results.

On competent G, eight-noun A versus A–F at rank7 gives native KL/removal ratios0.345→0.199, but effect RMSE only0.154→0.151. Thus the pooled effect improvement cannot be promoted to a substantial competent-task causal improvement. Four-noun A–F rank7 gives0.145 effect RMSE, similar to these alternatives; no best-fit selection is made.

On G, the A/B rank1 effect RMSE is1.127 (eight nouns) or1.063 (four), worse even than cross-term removal0.688, despite native KL/removal ratios0.596/0.558. No native replacement condition changes G's top token or binary correctness. This is a concrete counterexample to using unchanged answers or small native KL as a sufficient mechanism-validation criterion.

The matched16-discovery-endpoint contrast is four-noun A/B versus eight-noun A. At rank7, pooled native ratios improve0.233→0.153, while effect RMSE worsens0.127→0.161. On competent G the corresponding effect RMSE is0.154→0.223. Coverage does not improve all criteria, even with endpoint count held fixed; lexical composition also differs.

Copying gains remain modest. Eight-noun A rank7 native ratio is0.750 versus0.630 for A–F; exact-flip approximation KL0.001347→0.001118 and effect RMSE0.047→0.040. The exact number-flip effect itself is weak in copying, so these small absolute errors do not establish successful recovery of the copying mechanism.

### Implications and next discriminator

A single rank1 contextual write is not reliably sufficient across discovery distributions. Broader rank7 fits improve native distribution fidelity, but competent-task intervention accuracy can saturate or worsen. Therefore neither reconstruction rank nor native output fidelity should define a fundamental computational unit by itself.

Next diagnose why the A/B rank1 fit loses intervention fidelity: decompose discovery output covariance into between-template and within-template variation, inspect which PCs represent each, and measure per-PC intervention contributions or exact downstream sensitivity. Keep the number axis and current fits fixed. This distinguishes variance ordering from downstream amplification and motivates a causally weighted featurizer only if independently evaluated. Continue broader manifold/BSF-alternative work with separate geometry, native fidelity, intervention fidelity and reusable-computation criteria; the broad goal remains incomplete.

### Verification

1,856 full-model forwards as planned. Exact native replacement produces zero logit difference; compiled encoders match explicit projections to relative2.13e−15. All source/model/unit/output-fit hashes unchanged and all KL values finite. No natural or reserved-test evaluation; natural-document pool remains26,112. Full per-example outputs and competence-stratified analyses saved.
