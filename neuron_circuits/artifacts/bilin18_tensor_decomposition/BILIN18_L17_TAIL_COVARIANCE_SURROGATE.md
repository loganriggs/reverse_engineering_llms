

## H386 — Trace-matched isotropic tail covariance with consistent target fitting

Reuse H384’s 512-document panel and all39 response conditions, both frozen bases and ranks16/64. Partition the full recipient-probability unembedding covariance M into q C_head + t C_tail + q t (mu_head−mu_tail)(mu_head−mu_tail)^T, where head is top32 and q+t=1. Replace C_tail by alpha I with alpha=trace(C_tail)/1152. Cross exact/isotropic tail covariance with retaining/removing the between-group mean term. For every arm, G=V^T M V and b=delta^T M V come from the SAME surrogate metric. The isotropic RHS contribution is t alpha delta^T V; this is a metric replacement, not zero-target ridge regression.

Both the original top32 covariance and its target cross-moment remain. The isotropic-plus-between arm still uses the actual tail mean and trace, so it is not independent of all tail information. The test removes within-tail directional anisotropy while preserving total tail covariance trace. All arms use the complete target response and the same native full-vocabulary finite KL step safeguard.

|Basis|Head|Family|Exact+between|Exact−between|Isotropic+between|Isotropic−between|
|---|---|---|---:|---:|---:|---:|
|probability|0|actual|0.8599|0.8472|0.8157|0.8046|
|probability|0|moment|0.7478|0.7341|0.6766|0.6627|
|probability|0|donor|0.7479|0.7341|0.6773|0.6639|
|probability|1|actual|0.8012|0.7715|0.7426|0.7160|
|probability|1|moment|0.7901|0.7640|0.7294|0.7050|
|probability|1|donor|0.7919|0.7641|0.7312|0.7039|
|probability|6|actual|0.8158|0.7840|0.7565|0.7308|
|probability|6|moment|0.8352|0.7850|0.7666|0.7404|
|probability|6|donor|0.8396|0.7882|0.7710|0.7444|
|random|0|actual|0.8132|0.7918|0.7687|0.7159|
|random|0|moment|0.7287|0.7155|0.6726|0.6471|
|random|0|donor|0.7272|0.7132|0.6703|0.6423|
|random|1|actual|0.7737|0.7456|0.7131|0.6593|
|random|1|moment|0.7610|0.7372|0.6996|0.6562|
|random|1|donor|0.7638|0.7384|0.7032|0.6567|
|random|6|actual|0.7738|0.7377|0.6621|0.5841|
|random|6|moment|0.8056|0.7515|0.7462|0.6607|
|random|6|donor|0.8128|0.7583|0.7541|0.6728|

Frozen gates: {'isotropic_within5': False, 'between_exact5': False, 'between_isotropic5': False}. Isotropic replacement requires all36 exact-minus-isotropic upper bounds with between retained <5pp. The two between-term gates require all36 retained-minus-removed lower bounds >5pp under exact or isotropic tails, respectively.

|Contrast|Point range (all36)|Global range of simultaneous bounds|
|---|---:|---:|
|exact_minus_isotropic_with_between|1.54 to 11.17 pp|-0.30 to 13.01 pp|
|exact_minus_isotropic_without_between|2.29 to 15.35 pp|0.45 to 17.19 pp|
|between_effect_exact|0.88 to 5.44 pp|-0.96 to 7.28 pp|
|between_effect_isotropic|0.65 to 8.55 pp|-1.19 to 10.39 pp|

Audits: input hashes verified; exact partition agrees with direct full-probability Gram and RHS to maximum relative error 3.1e-14. Original metric finite grids replay H384 within 3.7e-10 absolute KL. All16/64 directions retained; maximum normal-equation relative residual 4.4e-12. Independent saved-array audit verifies baseline agreement, grid minima, positive isotropic scales and valid tail masses. Per-site coefficient norms, scales, masses, grid values and choices remain saved. Nonworsening versus baseline follows by construction from the zero step. Independent audit fields are added by this reporting script after the experiment runner.

Uncertainty: 4,000 paired recipient-document/control-draw bootstrap resamples, simultaneous bands over144 contrasts. Fixed bases and donor assignments; reused panel, not fresh confirmation. Figure displays rank64 point estimates. No semantic-unit, native parameter compression, or target-free coefficient claim follows directly.

All three frozen gates FAIL. With the between-group term retained, isotropic replacement loses 1.54–11.17 percentage points across the36 settings; the largest simultaneous upper loss is13.01 points, exceeding the five-point margin. Thirty of36 joint lower bounds exceed zero. When the between term is removed from both arms, exact tail covariance beats isotropic in all36 simultaneous intervals (point losses2.29–15.35 points; minimum lower0.45 points). These results retain the failed universal gates rather than redefining success after observing outcomes.

The isotropic replacement nevertheless preserves substantial recovery: rank64 exact+between72.72–85.99%, isotropic+between66.21–81.57%. Thus generic stabilization plus the actual head covariance and group means is useful, but insufficient for the prespecified approximation. This rules out this particular trace-matched isotropic surrogate as universally within five points; it does not rule out every regularizer or prove that the remaining anisotropy is context-specific or semantic.

The between-group mean term also contributes: its removal costs0.88–5.44 points under exact tails, and0.65–8.55 under isotropic tails. Positive joint lower bounds occur in23/36 and25/36 settings, respectively; neither universal five-point criterion passes. The term is a single outer product but can matter appreciably, and becomes more important in some settings when within-tail directionality is removed. Do not infer that a rank-one term is a semantic unit from this sensitivity alone.

Solver residuals and the exact covariance-partition audit pass. Maximum squared coefficient/response norm ratios are41.14 and24.83 across probability/random bases, far below H385's mixed-system maxima41,111/69,881. The failure of the isotropic approximation therefore persists without the extreme inconsistency introduced by mixing Gram and target cross-moment from different metrics. This distinguishes missing directional structure from that earlier mismatch artifact.

Hypothesis update: (1) top32 covariance plus scalar tail stabilization explains substantial recovery; supported descriptively; (2) tail anisotropy can be discarded at five-point accuracy universally; disfavored for this surrogate; (3) the between-group mean term alone supplies the missing structure; insufficient, since isotropic+between still loses up to11.17 points; (4) shared directional anisotropy might suffice; unresolved; (5) recipient-specific tail covariance is essential; not yet established by an isotropic comparison. The twenty-hypothesis list from H384 remains the broader framework.

Next discriminator: replace the tail covariance by the already frozen discovery-population covariance K, scaled to match each recipient tail trace. Use the same surrogate metric consistently for both G and RHS; retain the top32 covariance and between-group term. Compare this shared anisotropic surrogate to the exact and isotropic arms on this fixed panel, without target-dependent tuning. This separates generic anisotropy from context-specific tail orientation. If it succeeds, confirm on fresh documents before attempting target-free coefficient prediction. Future fresh panels must exclude5,824 complete-document hashes. Broad goal remains active.
