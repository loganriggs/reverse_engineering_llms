## H707 — Stable residual directions carry context beyond token identity

Compare all three seed pairs for H706's learned and random rotations at value0/residual1. Match256normalized decoder directions one-to-one using maximum absolute cosine assignment. Report decoder cosine, whitened rotation cosine and Top32active-support Jaccard. Random controls share the same training PCA span/anisotropy, making them more appropriate than unrelated ambient random directions for this repeatability question.

| Site | Rotation | Mean matched decoder cosine | Mean whitened cosine | Mean support Jaccard | Directions with cosine>0.9, out of256 |
|---|---|---:|---:|---:|---:|
|value0|random|0.234|0.147|0.081|0.0|
|value0|learned|0.605|0.585|0.230|83.7|
|residual1|random|0.344|0.110|0.070|0.0|
|residual1|learned|0.730|0.709|0.352|115.0|

Learned directions repeat substantially more than random rotations, especially residual1. This is partial repeatability, not uniqueness: average support overlap remains0.352for residual1and some directions match poorly. Matching uses decoder geometry only; support agreement is evaluated afterward. Anisotropy inflates random residual decoder cosine relative to whitened cosine, so both are retained.

Select eight seed7061directions per site by minimum(cosine×Jaccard) across its two other-seed matches, requiring at least32validation activations. Fit smoothed token-identity and position predictors from training supports, and evaluate validation Brier skill relative to a marginal activity-rate predictor. Brier skill measures predictive improvement, not fraction of semantic content. For repeatedly active training tokens, also measure within-token signed coefficient variance divided by total coefficient variance on the same subset.

| Selected directions | Mean token Brier skill | Mean position Brier skill | Mean within-token coefficient variance fraction |
|---|---:|---:|---:|
|value0|0.782|-0.000|0.000|
|residual1|0.170|0.018|0.626|

Value0behaves as the expected token-only control: token identity strongly predicts supports, repeated tokens never mix active/inactive status, and same-token coefficient variation is numerical zero. For residual1, token prediction is much weaker (mean0.170skill), position explains little (0.018), and within-token coefficient variation is substantial (mean0.626, range0.130–0.939). Repeated validation token types mix active/inactive states for61–84types per selected direction. These checks support context dependence beyond token identity, not semantic specificity or a task algorithm.

Top active tokens remain common punctuation/function words: periods, commas, newlines, 'the', 'and'. Counts are descriptive and not frequency-adjusted enrichment. Context dependence might include syntax, position interactions, scale changes or heterogeneous features. No manifold dimensionality or unsupervised task labels have been established. Scalar-direction results should not be conflated with earlier16dimensional BSF blocks, whose selection/model budgets differ.

Next use matched token-and-position context pairs to swap selected residual1coordinate contributions, with matched-norm controls and a full-residual donor reference. Measure downstream output-distribution changes and donorward transfer without equating either with task correctness. The dual encoder/decoder basis is oblique after unwhitening, so distinguish coordinate substitution from Euclidean projection onto a decoder direction. This tests whether stable context variation is causally useful before grouping directions into candidate manifolds or regime labels.

Audits: source hashes unchanged, finite matching metrics, training-only predictors, test evaluations0, whole-model forwards0, no new documents. Validation was reused for exploratory direction selection/descriptions, so these are not independent confirmation estimates. Broad goal active.
