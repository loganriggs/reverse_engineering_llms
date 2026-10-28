

## H353 — Does generic output confidence explain metric pairing?

Reuse H346 responses and metrics. Collect intact output entropy and top1 probability log odds. Standardize both features separately at each token position. Select the six nearest other documents at the same position, using no response outcomes. Transfer their physical-trace-normalized metrics to each recipient; compare own and random different-document metrics with the same equal-site output-Rayleigh statistic as H352. Neighbors are selected with replacement across recipients.

|Metric|Moment control/actual [95% interval]|Donor control/actual [95% interval]|
|---|---:|---:|
|own|1.878 [1.647, 2.125]|1.788 [1.579, 2.015]|
|matched|1.269 [1.172, 1.374]|1.211 [1.120, 1.308]|
|shuffled|1.252 [1.154, 1.356]|1.200 [1.114, 1.289]|

Own/matched ratios: moment1.479 [1.312,1.676], donor1.476 [1.305,1.671]. Confidence-matched and random-transfer point estimates are close; this is not an equivalence test. Both frozen gates pass: own/matched lower95%>1.2 for both controls, and confidence distance reduced by more than50%.

Matching audit: mean standardized distance 0.200 versus random 1.618, a87.66% reduction; worst matched distance 2.773; zero self matches. Matching is imperfect, particularly in sparse confidence regions. Donor identities, features and per-site response metrics are saved.

Hypothesis update: H352 candidate7 (entropy/top1 confidence accounts for pairing) is weakened. Candidate1 (context-specific output alignment) remains supported descriptively. This does not distinguish semantic coordination from unmeasured geometry, token identities, cap saturation or compiler artifacts. It is not evidence that all generic confidence variables have been controlled.

Uncertainty:4000 joint document resamples and independent global control-realization/neighbor-rank resamples. Neighbor selection is fixed during bootstrap; donor reuse and dependence are not reconstructed. Pointwise conditional intervals on reused H346 data. Matching on outputs is a diagnostic comparison, not randomized semantic intervention.

Next: validate the pairing effect for native MLP output responses, keeping the same payloads and confidence-matched donors. Avoid interpreting compiler-specific output geometry as a property of the whole MLP until this passes. Broad research goal remains active.
