

## H352 — Shared output geometry versus context pairing

Freeze H346 response coefficients. Normalize each output metric M by trace(M G⁻¹), where G=outputs@outputs.T, giving unit trace in an orthonormal basis of the physical compiler-output span. Compare own-context metric, a position-specific average of normalized H339 metrics, and six different-document cyclic metric assignments preserving position. This isolates metric shape from overall sensitivity scale. Earlier metrics use no H346 responses; evaluation responses and analysis questions are reused.

At each site measure coefficientᵀM coefficient divided by Euclidean response energy, and average sites/realizations equally. This is an output-metric diagnostic, not a realizable model intervention.

|Metric|Moment ratio [95% interval]|Donor ratio [95% interval]|
|---|---:|---:|
|own|1.878 [1.646, 2.130]|1.788 [1.585, 2.017]|
|earlier_average|1.290 [1.218, 1.366]|1.238 [1.175, 1.305]|
|shuffled|1.252 [1.157, 1.353]|1.200 [1.111, 1.288]|

Own/shuffled ratio is1.499 [1.326,1.704] for moment and1.490 [1.317,1.697] for donor controls. The pairing-amplification threshold passes. The config label pairing_required is too strong literally: a smaller gap remains after shuffling, so pairing amplifies rather than being necessary for any gap. The shared-gap gate requires earlier-average lower95%>1.2 for both; it fails because donor lower bound1.175. This does not show zero shared structure.

Audit: input hashes verified; physical metric trace error 4.44e-16. 4000 paired document and independent response-realization/metric-shift bootstrap draws; pointwise conditional intervals. Cyclic donor dependencies are not reconstructed. Metrics, response energies and bootstrap draws saved.

### Refreshed20-hypothesis tournament

These hypotheses overlap; ranking is qualitative, not posterior probability. Status uses evidence throughH352.

|#|Hypothesis|Current assessment|Discriminating experiment|
|---|---|---|---|
|1|Context-specific output alignment|Leading: own-metric gap exceeds shuffled|Transfer metric within matched confidence and residual-geometry strata|
|2|A shared output direction explains most of the gap|Partial: earlier mean preserves smaller gap|Fit shared low-rank metric on earlier panel; freeze and evaluate|
|3|RMS radial finite scaling explains the gap|Insufficient: removal reduces ratio20–22%|Use fixed-tangent radial dose response|
|4|Natural responses are preferentially radial|Contradicted on panel|Already tested: controls more radial|
|5|Radial correction cancels natural output effects|Contradicted in aggregate|Already tested: signed cross terms positive|
|6|Probability weighting alone explains amplification|Insufficient: probability/uniform factor~1.28|Match probability spectra while permuting token identities|
|7|Output confidence accounts for context pairing|Open, high priority|Stratify or match entropy/top1 probability before metric transfer|
|8|Token-specific logit geometry matters|Open, high priority|Contrast spectrum-matched probability permutations with intact token assignment|
|9|Unembedding anisotropy is sufficient|Partial at most|Uniform metric and frozen shared-output metric controls|
|10|Cap saturation drives the context effect|Open|Repeat factorial with cap derivative removed|
|11|Edit curvature drives amplification|Disfavored: H342 small quadratic contribution|Fresh exact tangent/quadratic comparison|
|12|A few exceptional documents drive the effect|Disfavored: H340 robustness, H345 most sites|Fresh robust quantiles and leave-high-energy-out analysis|
|13|Only changed context weights explain gain|Disfavored by H345/346|Already tested: equal-site normalized gap persists|
|14|Gaussian source controls miss essential higher moments|Open for magnitude; not sole effect|Empirical donor and covariance controls under matched context strata|
|15|Recipient-source semantic coordination matters|Plausible but unproven|Task-controlled donor swaps with independently annotated semantic roles|
|16|The compiler induces a spurious alignment|Open, high priority|Run native MLP Jacobian/output-response analog without compiler restriction|
|17|Head6 is an isolated special case|Open|Freeze same statistic across all heads with simultaneous inference|
|18|The effect is specific to final layer|Open|Repeat finite-output-path analysis across selected depths|
|19|Stable feature units underlie the local directions|Unproven|Cross-context direction transfer and basis-invariant subspace stability|
|20|Computation is distributed through context-dependent interactions|Consistent with native-factor and local results|Compare heldout fixed-unit versus conditional-interaction predictors at matched cost|

Priority shortlist: context pairing (#1), confidence versus token-specific output geometry (#7–8), compiler restriction (#16), and shared versus conditional representation (#2/#19/#20). Next run a confidence-matched metric transfer; it can distinguish generic confidence from more specific context alignment without relying on semantic labels. Then validate the surviving result using native MLP responses. Broad research goal remains active.
