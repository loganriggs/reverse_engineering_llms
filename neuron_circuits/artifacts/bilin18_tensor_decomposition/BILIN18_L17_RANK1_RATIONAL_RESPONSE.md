## H486 — Exact rational rank1 response and necessary nonlinear terms

Use the fixed calibration top1 source direction and the same512 H484/H485 receivers, both signed donor amplitudes. This is an exploratory native-model algebra/control experiment, not fresh confirmation or a learned replacement. Direct residual path remains fixed. All recovery denominators here are native rank1 intervention effects, not the full-source effects used in H485.

For unnormalized receiver p and fixed source direction b, intervene p−a b. The bias-free bilinear write is

w(a) = (N0 + a N1 + a² N2) / (d0 + a d1 + a² d2),

where N0=D[(Lp)⊙(Rp)], N1=−D[(Lp)⊙(Rb)+(Lb)⊙(Rp)], N2=D[(Lb)⊙(Rb)], d0=mean(p²)+eps, d1=−2mean(p b), and d2=mean(b²). The Down bias cancels in the response. Use the native float32 RMS epsilon even for float64 algebra.

Writing w0=N0/d0, V1=N1−w0 d1 and V2=N2−w0 d2 gives the exact response

Δw(a) = [a V1 + a² V2] / [d0 + a d1 + a² d2].

Thus at a fixed receiver the entire response curve along this source direction lies in the span of two output vectors. Those vectors vary across receivers: this is not a global rank2 output space or a discovered pair of semantic features. It follows from the architecture, and now has numerical verification. Constructing these vectors still uses the native weights and receiver state; no computational/storage saving is established by the identity alone.

| Approximation to native rank1 response | KL recovery −/+ | Raw-response recovery −/+ |
|---|---|---|
|exact|100.0000% / 100.0000%|100.0000% / 100.0000%|
|tangent|82.4703% / 84.3478%|98.0174% / 97.5240%|
|linear_numerator|82.8235% / 86.6826%|97.2362% / 96.8688%|
|frozen_norm_full|18.2250% / -46.3288%|63.8708% / 48.3653%|
|frozen_norm_linear|6.9033% / -47.7284%|61.0245% / 52.1610%|
|normalization_only|-105.0536% / -134.3297%|-91.1922% / -91.8772%|

Definitions: exact retains both numerator terms and the exact denominator. Tangent is a V1/d0, the true derivative at zero amplitude times a. Linear_numerator retains a V1 with the exact changed denominator, dropping a² V2. Frozen_norm_full keeps the bilinear numerator change (a N1+a² N2) but fixes the RMS-squared denominator to d0. Frozen_norm_linear also drops the quadratic numerator. Normalization_only changes the denominator of N0 without changing the numerator. These controls alter specified mechanisms; they are not equally trained predictors. Negative recovery means more KL error than predicting no intervention effect.

The exact formula matches float64 direct evaluation below1e−8 relative response error, and float32 native response below1e−4. Its finite-KL and raw recovery are effectively100%. The true tangent retains~97.5–98.0% of raw response energy but only82.47/84.35% of probability effects. The corrected linear numerator with exact denominator yields82.82/86.68% KL recovery, still below the exact response. The quadratic corrected term carries only2.76%/3.13% of exact squared-response energy, yet dropping it leaves substantial probability error. Cross terms mean these raw energies are not additive explained shares.

Freezing the normalization denominator yields18.23/−46.33% KL recovery even with the full bilinear numerator. Changing only the denominator also fails badly. Their separate failures show that the numerator response and normalization correction must be considered together; neither is an independently sufficient account. This does not assign additive causal percentages to the mechanisms. Relative denominator changes range from about−14.5% to+29.0% at negative sign and−14.0% to+25.1% at positive sign; the median changes are+1.36% and−.95%. Small typical denominator shifts do not imply negligible probability effects.

Hypothesis update: H485's mostly rank1 source does not imply an almost linear behavioral response. Finite-amplitude curvature and RMS-mediated coupling matter in this regime. The problem is more structured than an arbitrary map from quadratic input coordinates to output coefficients: a fixed source direction induces a receiver-linear cross operator, a fixed quadratic vector and an exact scalar normalization correction. This offers a more targeted data-based canonical form than selecting a global output-energy basis without those terms.

Indeed N1=M_b p for a fixed operator M_b=−D[diag(Rb)L+diag(Lb)R]. Equivalently,

Δw(a) = a M_b p/d(a) + a² N2/d(a) + [d0/d(a)−1]w0.

Next evaluate a normalization-aware structured approximation: retain the known scalar denominator, fixed N2 and baseline write w0 exactly, and compress only M_b p under the training receiver distribution. Fit a small data-weighted linear map with training-only coordinates, select rank on validation, and compare with omitting the exact normalization correction at matched cross-map rank. Audit storage/cost separately: w0 is already available only in response-isolation experiments and cannot be called a free full-layer replacement. Measure both rank1-target fidelity and remaining full-source error; preserve the source tail as a separate limitation. Earlier residual-scaling work provides related machinery, but the new comparison must explicitly isolate this fixed-source cross operator and finite-amplitude normalization.

Audits: exact float64 identity, float32 response replay, H485 rank1-effect replay and finite arrays pass. Save receiver-specific V1/V2, denominator coefficients, signed amplitudes, per-case errors/effects and numerator cross terms. No new documents, fitting or source calibration; total16832 tracked documents unchanged. No confirmatory intervals or learned-performance gates are claimed. Broader canonical comparisons including observed-token CE and transfer across heads/layers remain open.
