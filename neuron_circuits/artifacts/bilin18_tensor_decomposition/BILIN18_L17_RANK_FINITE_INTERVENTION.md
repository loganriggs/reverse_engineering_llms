

## H366 — Two weight directions predict pairing but fail finite sufficiency

Reuse H365 native MLP-only response vectors for heads 0, 1 and 6 on its 128-document panel, with actual, six moment and six donor conditions. These heads were selected from previous results. This is an exploratory mechanism test on an existing panel, not another independent confirmation. Freeze H364 top-2/top-16 weight eigenvectors and the first two columns of its fixed random basis before measuring finite outcomes.

For intact final residual s and native MLP response d, compute a=RMS(s), b=RMS(s+d), and Δ=b−a. Keep2 uses a+P₂Δ; Remove2 uses a+(I−P₂)Δ; Keep16 and Random2 are corresponding projections. Feed these activations directly through unembedding and the logit cap, without another RMS normalization. These are off-manifold post-normalization patches, not input edits or a replacement MLP. The cached native response excludes the direct attention-head residual change.

Score recovery = 1 − KL(p(b) || p(patch))/KL(p(b) || p(a)). Thus 1 is exact recovery, 0 matches doing nothing, and negative values mean a worse approximation than doing nothing. KL is in nats. This KL orientation differs from earlier intact-to-edited behavioral reports.

|Head|Family|Baseline KL|Keep2 recovery|Remove2 recovery|Keep16 recovery|Random2 recovery|
|---|---|---:|---:|---:|---:|---:|
|0|actual|0.000268215|0.266|-0.002|0.324|-0.001|
|0|moment|0.00012715|-0.097|0.502|-0.112|-0.002|
|0|donor|0.000122822|-0.080|0.553|-0.090|-0.002|
|1|actual|0.00127813|-0.064|0.235|-0.191|-0.006|
|1|moment|0.00115501|-0.069|0.292|-0.201|-0.003|
|1|donor|0.00120531|-0.085|0.287|-0.219|-0.005|
|6|actual|0.00200553|-0.142|0.015|-0.028|-0.019|
|6|moment|0.00549674|-0.008|-0.338|0.074|-0.001|
|6|donor|0.00575584|0.019|-0.312|0.087|-0.003|

All 36 recovery estimates have simultaneous bootstrap bands of ±0.299. Resample 128 recipient documents and the six control realizations 4,000 times; fixed donor assignments remain conditional. Every Keep2 upper bound is below 0.8. All frozen gates fail: {'rank2_sufficient': False, 'rank2_necessary': False, 'rank16_sufficient': False, 'rank2_orientation': False}. The uniform necessity threshold also fails, but that does not mean the two axes are irrelevant at every head or condition. Negative values and head differences should not be collapsed to a single necessity claim.

Interpretation: a low-rank metric can predict a relative, aggregated context-pairing statistic without its eigenspace carrying most finite probability changes. The removed and retained components need not have additive effects; Euclidean orthogonality does not imply probability-metric orthogonality. Off-manifold patching and probability-weighted anisotropy are live explanations. H365’s successful predictor remains valid within its narrower scope.

Audits: all input artifact/corpus SHA256 checks pass; basis orthogonality passes; all KL values finite and nonnegative within tolerance; the full-target positive control has zero KL. Baseline KL ranges from 0.000123 to 0.00576 nats. Baseline top-1 agreement with target is already 97.5–99.2%, so token agreement alone is an insensitive success metric. Save every site-level KL and top-1 comparison, plus bootstrap recovery arrays. Native weights unchanged.

### Refreshed hypothesis tournament

|#|Hypothesis|Current standing and discriminating experiment|
|---|---|---|
|1|Two axes carry a sufficient finite output program|Disfavored by H366; all Keep2 upper bands below 80% recovery.|
|2|Two axes predict a relative statistic while broad directions carry behavior|Leading; test variance of pairing numerator/denominator separately across rank.|
|3|Rank-2 and complementary components cancel in the probability metric|Leading; decompose local Fisher energy into both self terms and signed cross term.|
|4|Finite curvature creates the failed projection result|Live; repeat normalized-delta patches at 1/4 and 1/16 amplitude and compare the Fisher limit.|
|5|Uniform vocabulary covariance selects directions irrelevant to likely tokens|Leading; compare probability-weighted covariance spectra with frozen weight axes.|
|6|Residual radial projection is the major context-dependent modulator|Supported by H361; test geometry at new layers and panels.|
|7|Radial alignment alone explains pairing|Disfavored by H362; isotropic predictor got the sign pattern wrong.|
|8|Off-manifold activation norms cause the finite failure|Live; compare norm-matched patches, explicitly separating direction and norm changes.|
|9|The logit cap changes which eigenvectors matter|Live; compare cap-aware versus cap-free derivative accounting on the same responses.|
|10|A few outlier documents drive poor recovery|Live; inspect document KL contributions and leave-top-documents-out curves.|
|11|Natural and artificial responses occupy different complementary directions|Live; compare actual versus control Fisher cross terms and principal-angle distributions.|
|12|Head 6 is a special cancellation regime|Plausible; H366 removal worsens control approximation; replicate beyond selected heads.|
|13|Head 0 contains a modest rank-2 behavioral component|Tentative point estimate only; its actual Keep2 recovery is 27%, with joint interval including zero.|
|14|Heads differ mainly in probability confidence|Weakened by confidence matching; test richer probability-spectrum matching.|
|15|The two axes encode frequency/common-token directions|Unresolved semantic interpretation; inspect weight loadings against token frequency without assuming labels.|
|16|Rank 2 is a regularizer for a misspecified downstream metric|Plausible; it predicts pairing better than full uniform covariance but fails sufficiency.|
|17|Other low-dimensional axes better preserve finite responses|Open; fit on discovery only, validate probability-metric subspaces on fresh documents.|
|18|A context-dependent small subspace works where a fixed subspace fails|Open; compare context-conditioned subspaces with parameter-matched fixed controls.|
|19|Native computation needs broadly distributed bilinear factors|Supported by earlier factor-compression failures; do not infer this solely from H366 output geometry.|
|20|The mechanism is depth-specific rather than a fundamental unit|Open; repeat the frozen diagnostic at additional layers before generalizing.|

Next highest-information experiment: decompose the local probability-metric energy of retained and complementary post-RMS deltas, and compare finite patches across amplitude. This distinguishes cross-component cancellation from finite curvature before searching for semantic labels or fitting more subspaces. Broader goal remains active.
