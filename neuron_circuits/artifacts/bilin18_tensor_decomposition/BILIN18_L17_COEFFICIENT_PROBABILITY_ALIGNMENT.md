

## H382 — Probability-to-token alignment matters beyond concentration

Keep native finite target/scoring and the eight-step safeguard unchanged. Fit rank16/64 coefficients in the frozen probability and first-random bases using three metrics, all without the cap derivative: the recipient’s own probability distribution (cached H381), its nearest other-document same-position donor matched in standardized entropy/logit-top1 space, or its own probability values reassigned to token identities by one fixed vocabulary permutation. The last control preserves the entire probability spectrum exactly but is not generally a realizable natural context. All methods still use the complete native response.

Matching reduces mean standardized feature distance from1.611 for the cyclic other-document control to0.0475 (97.05% reduction); maximum nearest-donor distance is0.591. Only3.61% of matched donors share the recipient’s most likely token. The permutation preserves entropy within1.8e-15nats. Thus donor confidence is closely matched but not identical; the permutation supplies an exact spectrum control. No response outcome enters matching or permutation selection.

|Basis|Head|Family|Own64|Matched-donor64|Spectrum-permuted64|
|---|---|---|---:|---:|---:|
|probability|0|actual|0.860|0.529|0.409|
|probability|0|moment|0.744|0.266|0.166|
|probability|0|donor|0.751|0.282|0.176|
|probability|1|actual|0.797|0.407|0.227|
|probability|1|moment|0.781|0.360|0.211|
|probability|1|donor|0.782|0.363|0.211|
|probability|6|actual|0.820|0.498|0.253|
|probability|6|moment|0.840|0.441|0.334|
|probability|6|donor|0.842|0.448|0.343|
|random|0|actual|0.809|0.405|0.303|
|random|0|moment|0.722|0.199|0.112|
|random|0|donor|0.728|0.208|0.118|
|random|1|actual|0.765|0.323|0.189|
|random|1|moment|0.746|0.268|0.163|
|random|1|donor|0.749|0.271|0.165|
|random|6|actual|0.775|0.352|0.175|
|random|6|moment|0.815|0.315|0.249|
|random|6|donor|0.816|0.317|0.253|

All frozen gates pass: {'confidence_balance': True, 'matched_alignment_gain': True, 'permuted_alignment_gain': True}. Across both bases, ranks and all9 head/family groups, own probabilities beat both controls by more than10pp in every joint lower bound. The smallest lower bound is19.36pp against matched donors and30.55pp against permutation. Across rank64 groups, own recovery is72.2–86.0%, matched-donor recovery19.9–52.9%, and spectrum-permuted recovery11.2–40.9%. Complete rank16 results are saved as well.

Interpretation: concentration is not sufficient for high oracle recovery. The fitting metric must align probability mass with the recipient’s relevant token-output directions. The natural donor control and exact spectrum control support this beyond the entropy association and temperature intervention. However token reassignment also changes unembedding norms/correlations and semantic relations, so this is not a clean intervention on semantics alone or proof of semantic feature units. It shows specificity of the probability-to-output-coordinate alignment in this fitting procedure.

Both controls receive native-target finite step selection, which can partially rescue poor fits; the large losses remain despite this common oracle safeguard. The finding therefore does not supply target-free coefficients or a compressed native MLP. High recovery in arbitrary spans appears to rely on fitting the right context-specific token constraints, not merely having a low-entropy metric.

Audits: source artifact hashes verified; donor indices exclude the same document and preserve position. All16/64 Gram directions retained; normal-equation residuals below2e-12; local fits do not exceed Euclidean projection error under their own fitting metric. Finite positivity/finite-value checks pass. Save matching features, donor indices, fixed permutation, per-site grid scores, choices and paired bootstrap arrays. Native weights unchanged.

Uncertainty:4,000 paired recipient-document/control-draw resamples; joint72 bands cover own-minus-control contrasts across both bases, ranks, controls and head/family groups. Fixed donor matching, empirical payload-donor assignments and one vocabulary permutation; no rebuilding of matching or random-orientation population uncertainty. Reused512-document panel. Figure shows point estimates, not confidence intervals.

Hypothesis update: both output concentration and recipient-specific token alignment matter for the oracle. This sharpens the alternative to fixed-feature interpretations: moderate-dimensional arbitrary spans can satisfy a context-selected set of likely-token constraints, but the coefficient selection must know which constraints matter. The actual generation of those coefficients from model inputs remains unresolved.

Next discriminator: approximate the recipient fitting metric with a prespecified number of its most likely tokens, keeping the full native distribution for evaluation. Measure how many token constraints are needed to recover the own-metric oracle result, with rank-deficiency audits and the same safeguard. Follow any selected approximation with a fresh-document confirmation before treating it as a transferable mechanism. Broader goal remains active.
