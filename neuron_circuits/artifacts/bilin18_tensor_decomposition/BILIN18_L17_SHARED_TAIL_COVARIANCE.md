

## H387 — Frozen shared anisotropy as a tail covariance surrogate

Keep H386’s exact top32 conditional covariance and head/tail mean outer product. Compare exact tail covariance, trace-matched isotropic covariance, and trace-matched frozen shared covariance. Shared tail surrogate = trace(C_tail)/trace(K) × K, where K is H373’s discovery-average within-context unembedding covariance, frozen from the older H365 panel. It is a full-probability covariance template, not a fitted tail-only template. Trace scaling uses the recipient tail, without response-dependent tuning. Both G and b use the same surrogate metric.

Reuse the H384512-document panel with39 response conditions, both frozen bases, ranks16/64, and native full-vocabulary finite KL scoring with the unchanged eight-step safeguard. Actual recipient tail mean and trace remain available. All coefficients and step choices still require the target response.

|Basis|Head|Family|Exact64|Isotropic64|Shared64|
|---|---|---|---:|---:|---:|
|probability|0|actual|0.8599|0.8157|0.8256|
|probability|0|moment|0.7478|0.6766|0.6900|
|probability|0|donor|0.7479|0.6773|0.6903|
|probability|1|actual|0.8012|0.7426|0.7487|
|probability|1|moment|0.7901|0.7294|0.7344|
|probability|1|donor|0.7919|0.7312|0.7363|
|probability|6|actual|0.8158|0.7565|0.7621|
|probability|6|moment|0.8352|0.7666|0.7753|
|probability|6|donor|0.8396|0.7710|0.7796|
|random|0|actual|0.8132|0.7687|0.7767|
|random|0|moment|0.7287|0.6726|0.6759|
|random|0|donor|0.7272|0.6703|0.6734|
|random|1|actual|0.7737|0.7131|0.7278|
|random|1|moment|0.7610|0.6996|0.7116|
|random|1|donor|0.7638|0.7032|0.7155|
|random|6|actual|0.7738|0.6621|0.7235|
|random|6|moment|0.8056|0.7462|0.7631|
|random|6|donor|0.8128|0.7541|0.7712|

Frozen gates: {'shared_within5': False, 'shared_beats_isotropic': False}. Shared approximation requires every exact-minus-shared joint upper bound <5pp. Superiority requires every shared-minus-isotropic joint lower bound >0. Each applies to all36 basis/head/family/rank settings.

|Contrast|Point range (all36)|Global range of simultaneous bounds|
|---|---:|---:|
|exact_minus_shared|1.44 to 6.00 pp|0.12 to 7.32 pp|
|shared_minus_isotropic|-0.74 to 6.13 pp|-2.06 to 7.46 pp|

Audits: input hashes verified; frozen K positive definite. Exact covariance partition max relative error 3.1e-14. Exact full-metric finite grid replays H384 within 3.7e-10 absolute KL; isotropic grid replays H386 within 0. All16/64 Gram directions retained; normal-equation maximum relative residual 4.4e-12. Independent saved-array audit confirms baseline agreement and exact minima over saved grids. All scales positive; norms, scales, mass, scores and step choices retained. Independent audit fields are added by this reporting script.

Uncertainty: 4,000 paired recipient-document/control-draw resamples, joint72 bands over both contrasts. Fixed bases, donor assignments and one frozen covariance template. Reused panel, not fresh confirmation. Figure shows rank64 point estimates; JSON includes both ranks and intervals. A full-covariance template failure does not rule out all shared tail-specific templates or prove context-specific semantic units.

Both frozen gates FAIL. The shared template loses1.44–6.00 percentage points relative to the exact recipient tail, with all36 simultaneous lower loss bounds positive (minimum0.12pp); the largest upper loss is7.32pp. Only11/36 upper bounds fall below the five-point margin. Shared-minus-isotropic effects range−0.74 to+6.13pp:24/36 point estimates favor shared, but only5/36 joint lower bounds are positive. Thus neither universal five-point approximation nor universal superiority over isotropic is established.

At rank64, exact recovery is72.72–85.99%, isotropic66.21–81.57%, and shared67.34–82.56%. Shared anisotropy helps some settings, including some larger isotropic deficits, but does not close the gap reliably. The exact and isotropic arms reproduce the prior runs; the isotropic finite grids are identical, so these differences are attributable to the new covariance substitution rather than replay drift. Positive definite metrics, retained ranks, and normal-equation audits pass.

Interpretation: the scalar-tail-stabilization explanation is incomplete, and simply inserting the already available global covariance is also insufficient at the declared margin. But this is not a demonstration that every usable tail covariance must be context-specific: K was averaged over FULL probability distributions, whereas the replaced object is conditional covariance below the top32 tokens. That mismatch is a concrete alternative explanation. Actual recipient tail mean and trace are still retained, and the coefficient fit still observes the full response.

Hypothesis update: (1) this frozen full-distribution shared covariance is a sufficient generic tail shape; disfavored at five-point accuracy; (2) shared anisotropy is always better than isotropic; not supported; (3) an appropriately trained tail-only covariance could explain the remaining gap; unresolved and now the highest-priority control; (4) context-specific tail orientation matters beyond a matched population template; plausible but not established; (5) these output-metric substitutions reveal fixed semantic units; unsupported.

Next discriminator: construct a shared tail-only conditional covariance on the older, disjoint discovery documents, using the same top32 exclusion rule. Freeze the mean conditional covariance before evaluating the current panel, then match its trace to each recipient tail and use it consistently for both G and b. Compare against exact, isotropic, and the already tested full-distribution template. This removes the known mismatch in what population statistic is averaged. No target-response fitting or hyperparameter sweep is needed to construct the covariance. A successful surrogate requires fresh-document confirmation; eventual target-free response/coefficient prediction remains necessary for the broad computational-unit goal. Future fresh panels must exclude5,824 complete-document hashes. Goal remains active.
