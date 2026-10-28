## H481 — Native coordinates versus omitted response remainder

Post-hoc factorial diagnosis on the same512 H479 receivers and two signs. Freeze the learned r256 input-to-coefficient predictor and16 output directions. Let u be the native response, P its Euclidean output-span projector, a=Pu its true in-span component, r=u−a its orthogonal remainder, and v the learned response. Let d be the in-span Fisher compensation from H480, so a+d is the locally probability-optimal span projection. Compare v, a, v+r, a+r, v+d and a+d. Every arm except v uses native intervention information and is oracle-only. Native intact baseline is retained to isolate response prediction; the comparison is not full-layer replacement or fresh confirmation.

| Response | KL recovery −/+ | Raw recovery −/+ |
|---|---|---|
|learned|77.31% / 72.83%|89.81% / 89.24%|
|true_inspan|79.39% / 75.15%|97.94% / 97.90%|
|learned_plus_remainder|89.93% / 88.96%|91.88% / 91.34%|
|true_inspan_plus_remainder|100.00% / 100.00%|100.00% / 100.00%|
|learned_plus_compensation|82.90% / 79.99%|58.96% / 55.45%|
|true_inspan_plus_compensation|92.14% / 91.60%|78.74% / 76.66%|

The true orthogonal remainder carries only2.06%/2.10% of total squared response norm. Restoring it to the learned prediction raises KL recovery77.31/72.83% to89.93/88.96%, removing about56%/59% of the learned predictor's finite KL error. In contrast, replacing the learned coefficients with the true Euclidean in-span coefficients, without the remainder, yields79.39/75.15%: a much smaller gain. Restoring both native components reconstructs the true response and recovers100% within measured numerical error; that is a control, not a learned result.

Adding the in-span Fisher compensation to the learned prediction raises KL recovery to82.90/79.99%, less than restoring the actual remainder. Combining that compensation with true raw coefficients yields92.14/91.60%, replaying H480's span oracle. Thus behavior-preserving compression can emulate much of the omitted component's probability effect inside the original span, but its coefficients differ from the native raw coordinates. The learned predictor has errors in these compensating coefficients, and prediction of native in-span coordinates alone does not resolve the omission.

### Local decomposition and interactions

Let e=v−a. Under the baseline Fisher metric, the original error is ||e−r||_F² = ||e||_F² + ||r||_F² −2<e,r>_F. Relative to native effect energy, the components are:

| Sign | In-span coefficient error | Omitted remainder | Cross inner product | Original error |
|---|---:|---:|---:|---:|
|negative|0.0943|0.1777|0.0348|0.2025|
|positive|0.1144|0.2604|0.0416|0.2916|

Although Euclidean-orthogonal to the retained span, the remainder is not Fisher-orthogonal. Its local error energy is17.77%/26.04% of the native probability-effect energy. Positive cross inner products partially cancel coefficient and omission errors; treating these as independently additive losses would be wrong. The compensated remainder r−d is Fisher-orthogonal to the span, so ||e−(r−d)||_F²=||e||_F²+||r−d||_F². This identity is independently verified for every case. Local identities are not asserted to hold for finite KL through the nonlinear readout.

The compensation identity G(c_F−c_raw)=B F r, Euclidean remainder orthogonality, and Fisher residual orthogonality all pass numerical checks. Original, raw projection and full-span compensation outcomes replay H480. Full native-response reconstruction has pooled KL error below1e−5 of native effect, with observed recovery effectively1. Arrays and scalar summaries are finite. Per-case finite errors, local errors, denominators and cross terms are archived. No confidence intervals or post-hoc success gates are claimed; the panel and oracle targets have been reused. Sources, calibration and15808 tracked documents unchanged.

Interpretation: high raw-energy coverage is insufficient for selecting behaviorally faithful computational units. A small-norm omitted write can have substantial output-probability consequences. The fixed span can mimic many of those consequences through context-dependent compensating coordinates, but those coordinates should not be mislabeled as native feature amplitudes. These findings concern this layer, source family, intervention size and output metric, not a model-wide ontology.

Next test a learnable alternative to compensation: derive a few output directions from the native training-response remainder, keeping the input representation and training/validation split fixed. Compare adding four training-residual principal directions with four frozen random orthogonal directions and the original16-dimensional span, fit the same Fisher coefficient objective, and select regularization on validation. Reserve a fresh confirmation panel before fitting. Equal-width added-direction controls distinguish a useful remainder basis from generic extra capacity. Residual PCA is a hypothesis about useful directions, not assumed optimal for KL. If it fails, receiver-dependent compensation or a readout-weighted direction selection remains plausible. CE objectives, source rank1/tail behavior and transfer across heads/layers remain open.
