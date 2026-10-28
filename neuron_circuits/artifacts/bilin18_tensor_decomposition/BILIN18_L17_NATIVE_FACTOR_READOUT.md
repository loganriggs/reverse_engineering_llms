

## H323 — Native multiplicative factors versus rotated factors

**H323 additional interpretation and audit.** Native factors support90.69%/89.97% fitted response fidelity versus0.67%/0.71% for the rotated control. This supports learned factor alignment, not individual semantic labels or compactness. Independent512-reference algebra audit: explicit features match native cross responses to2.21e-14 relative error; subtracting compiler cross terms matches cached residual labels to1.17e-5 (float32 construction). Saved factors match loaded weights exactly. The config describes the rotation as Haar, but Gaussian QR was used without sign correction; the accurate description is a QR-generated random orthogonal control. No Haar-distribution claim is made.

Use the same6144 labeled references as H321–322. For each native hidden channel j, represent the missing interaction by (core_z·L_j)(complement_z·R_j)+(core_z·R_j)(complement_z·L_j), minus the analogous intact-state product. There are4608 such features. Fit a shared full-output linear readout. Control: right-multiply both native factor matrices by the same QR-generated random orthogonal matrix, preserving their full pairwise Gram matrix and feature count while disrupting input alignment. One rotation seed is tested.

Each variant independently normalizes mean reference feature energy and selects one common ridge from0.0001,0.001,0.01,0.1,1,10,100 by worst-family calibration fidelity on documents768:896. Evaluate896:1024. All query documents have prior diagnostic use; no fresh confirmation. Reference documents0:768 and all6144 assignments remain fixed.

| Target | Native ridge | Native fidelity | Rotated ridge | Rotated fidelity | Gain (pp), conditional paired95% | Native Down witness |
|---|---:|---:|---:|---:|---:|---:|
|actual|0.001|90.69%|100.0|0.67%|90.02 [89.77,90.29]|84.790%|
|angle|0.001|89.97%|100.0|0.71%|89.26 [88.41,89.86]|86.844%|

Locked gates: native_gain10: PASS, native50: PASS. native_gain10 requires>=10pp gain and positive conditional interval in both families; native50 requires>=50% fitted native fidelity in both.

**Interpretation.** The native Down witness uses known native output weights, not a learned readout; it tests representability and the size of the omitted compiler correction. The fitted models use only reference response labels for output fitting. Better native-factor transfer would establish a useful weight-informed coordinate prior, not semantic neuron labels or a unique ontology. Joint input rotation preserves raw factor Gram structure but changes projected energy and correlations, so the contrast does not isolate semantics.

**Cost and limits.** Each fitted readout stores4608×1152=5,308,416 output coefficients. Storing both full factor matrices adds10,616,832 scalars, totaling15,925,248 before core, partner, mean, and normalization. This is essentially the native MLP weight count, not compression. The point is to test whether shared response recovery becomes possible in the learned multiplicative coordinates before pursuing sparse or lower-rank factors. Conditional bootstrap uses128 evaluation documents and omits reference/selection/adaptive research uncertainty.

**Audit.** Frozen reference and response hashes passed. Both selected primal normal-equation residuals are below1e-7. All coefficients, factors, geometry, normalization and selected penalties are saved; per-document errors and complete penalty curves are retained. Native weights unchanged.

**Next discriminator.** If native alignment helps, test reduced subsets or structured combinations of factors with fresh cross-source confirmation and full storage accounting. If it fails despite a strong known-weight witness, separate sample conditioning from representational failure. Broad research goal remains active.
