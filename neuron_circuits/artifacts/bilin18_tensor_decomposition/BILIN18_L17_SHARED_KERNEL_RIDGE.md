

## H318 — Ridge audit of the shared mixed-head predictor

Hold H317 mixed assignment,3072 reference examples, interaction descriptors, target labels and kernel normalization fixed. Vary only ridge over.0001/.001/.01/.1/1/10/100. First128 of the original256 discovery-validation documents select one common ridge by maximum worst-family pooled fidelity; remaining128 evaluate. Both halves have been reused in prior research, so neither is fresh. No H315/H317 diagnostic panel outcomes enter selection. Original.01 remains an explicit control.

| Ridge | Actual calibration | Actual evaluation | Angle calibration | Angle evaluation |
|---|---:|---:|---:|---:|
|0.0001|-4.247%|-5.700%|-4.689%|-6.008%|
|0.001|-3.444%|-4.807%|-4.171%|-5.434%|
|0.01|-1.714%|-2.884%|-3.237%|-4.345%|
|0.1|0.943%|0.090%|-1.789%|-2.599%|
|1.0|3.378%|2.955%|0.457%|0.108%|
|10.0|2.152%|2.041%|1.090%|1.151%|
|100.0|0.412%|0.401%|0.404%|0.455%|

Selected shared ridge: **10.0**.

- actual evaluation gain over.01: 4.925pp, paired document95%[3.809,6.073].
- angle evaluation gain over.01: 5.495pp, paired document95%[4.150,6.949].

Locked gates: shared10: FAIL, gain5: FAIL. Shared10 requires>=10% evaluation fidelity in both families; gain5 requires>=5pp improvement over.01 in both. Paired2000-document-resample intervals are descriptive conditional uncertainty on the128 evaluation documents, not fresh replication.

The worst-family calibration rule selects ridge10,1000× the inherited.01. Evaluation fidelity improves from−2.884% to2.041% actual and−4.345% to1.151% angle. Gains4.925/5.495pp have positive conditional paired intervals, but the strict5pp-in-both gate fails because actual is below5pp. Thus regularization explains the negative overshoot but not the large specialist/shared gap. The selected model still misses~98–99% of this residual cross-response energy.

**Interpretation limits.** This isolates a regularization change within the existing sampled mixed feature span. A ridge sweep cannot establish adequate reference coverage, a correct population optimum or whether a compact shared predictor exists. The exact bilinear polynomial remains shared mathematically; fixed dual references may span only a small part of its relevant input space. Positive fidelity after shrinkage would repair overshoot without necessarily recovering substantial computation. Low performance after a broad grid does not prove intrinsically head-specific rules.

**Audit.** Source response and frozen mixed-model hashes verified. Reconstructed.01 coefficients from the eigensolve match the saved H317 Cholesky coefficients to relative<1e-6. Kernel is PSD within numerical tolerance; negative roundoff eigenvalues clamped to zero. Common-ridge selection independently replayed from calibration scores; no evaluation-based model choice. Selected coefficients and descriptors saved for reproducibility. All jobs terminal; native weights unchanged.

**Next discriminator.** Hold the selected regularization rule fixed and vary mixed reference coverage/sample count, with matched-budget specialist controls. Analyze how much of each target feature lies in the span of training interaction features, independently of output fitting. That geometric support diagnostic can distinguish poorly covered input interactions from readout optimization failure and guide a more economical primal factorization. Reserve fresh documents for a meaningfully improved frozen candidate. Broad research goal active.
