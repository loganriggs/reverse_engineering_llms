## H489 — Structured computation over multiple source coordinates

Extend the fixed-source rational formula to the first1/2/4 source axes from the old disjoint calibration pool. For source projection h_k=B_k a, retain exact scalar RMS denominator, source-quadratic core Q_ij=D[(Lb_i)⊙(Rb_j)], and native baseline write. Each source axis has receiver-linear cross map M_i=−D[diag(Rb_i)L+diag(Lb_i)R]. The signed source coefficients enter the summed cross response and quadratic core; both ordered core indices are included, so cross-products are counted correctly.

For each k, construct the same1024 training receivers' summed cross contributions, with32 fixed donor assignments and both signs. Fit a shared32-dimensional output SVD to those contributions. Compile a separate1152x32 read for each source axis, a shared32x1152 write, source basis and quadratic core. Thus cross-output rank stays32, but increasing source rank increases parameter count and input interaction capacity. All source basis and cross-map fitting is training/calibration-only. No new feature labels or unique canonical basis are claimed.

Validation uses the same256 receivers and64 donors. Choose the smallest source rank with pooled full-source KL recovery>.95 for both signs, otherwise select4 and flag criterion unmet. No rank meets that validation criterion; source4 is the predeclared fallback, not a successful validation selection. Freeze all maps and selection before collecting new512-receiver confirmation outcomes.

Independent audit rehashes17344 excluded documents and512 new documents, verifies their129-token windows and finds zero complete-document overlap. Total tracked documents17856. Calibration/source families stay fixed; freshness is within the continuation, not a model-training holdout.

| Source rank | Compressed full-source KL recovery −/+ | Exact-cross full-source KL recovery −/+ | Compressed raw recovery −/+ |
|---|---|---|---|
|1|91.665% / 90.977%|95.578% / 94.943%|98.267% / 98.250%|
|2|93.126% / 92.733%|96.878% / 96.433%|98.716% / 98.673%|
|4|94.318% / 94.223%|98.461% / 98.265%|99.229% / 99.160%|

Exact-cross retains the projected-source computation without cross-map compression. It diagnoses source truncation, and is not a compact deployment result. Every recovery uses the full-source native effect as denominator. Native intact write remains supplied exactly for response isolation; no standalone full-layer replacement or CE-preservation claim.

| Predeclared primary statistic | Estimate | Joint95% interval | Threshold | Pass |
|---|---:|---|---:|---|
|absolute_negative|0.9432|[0.9315, 0.9548]|0.95|False|
|source_gain_negative|0.0265|[0.0174, 0.0357]|0|True|
|absolute_positive|0.9422|[0.9294, 0.9551]|0.95|False|
|source_gain_positive|0.0325|[0.0205, 0.0444]|0|True|

Both source-expansion gain gates pass: source4−source1 is2.65/3.25pp with joint95% intervals[1.74,3.57]/[2.05,4.44]pp. Both absolute>.95 gates fail; source4 point recoveries94.32/94.22% are themselves below95%. This is a confirmed benefit from the tested multi-coordinate structured extension, not attainment of the absolute fidelity target or proof that source rank4 is sufficient universally.

Exact-cross source4 reaches98.46/98.27%, versus compressed94.32/94.22%. That roughly4pp gap is larger than the remaining source-projection error under the exact-cross control. It identifies cross-map compression as the more immediate limitation in this conditional program. These differences are comparisons of errors against one target, not independent additive causal shares; cancellation could change after modifying either approximation. Source-rank gains also include the cost of additional per-source read maps, so this is not equal-total-parameter efficiency evidence.

Bootstrap uses4000 paired within-donor resamples of eight receiving contexts for each64 fixed donors. Four simultaneous standardized bands condition on source calibration, fitted maps, selection and donor pool. They do not include training-sample or source-pool uncertainty.

At k4/cross32, read tensors contain147456 scalars, shared write36864, source basis4608, and the full ordered quadratic core18432:207360 model-array scalars, plus an explicit epsilon constant in an inference implementation. The current archive contains all three candidate models, not only the selected one. This boundary excludes the native baseline computation and bias subtraction. Symmetric-core storage could be reduced but is not claimed for the saved representation; no runtime speedup is measured.

Audits: source bases orthonormal below1e−10; exact multi-source rational deltas reproduce native projected-source deltas below1e−4 relative error on validation and confirmation; all arrays finite; model/selection files saved before confirmation collection; independent document/window audit passes. Source rank1 validation outputs reproduce the preceding structured formula numerically. Compiled source bases, per-axis reads, shared writes, quadratic cores and per-case confirmation errors are archived. No process remains active.

Next hold source rank4 fixed and vary the shared cross-output rank32/64/128, using the same training distribution and exact normalization/core terms. This directly tests the diagnosed cross-map bottleneck rather than adding more source axes. Select the smallest rank meeting a frozen validation full-source fidelity criterion, retain a flagged fallback, then confirm on newly reserved documents. Compare costs and gains at a matched source rank. This can establish a compact conditional computational description; interpreting its source/receiver factors and removing native-baseline dependence remain separate work. Observed-token CE objectives and broader-head/layer transfer remain open.
