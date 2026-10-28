## H523 — Conditional stability of independently calibrated source maps

Four prescribed random complementary1,536/1,536 partitions of the pooled3,072 calibration sources (seed20280106) generate eight fits per method/rank. Fit full-covariance PCA and Fisher maps at ranks36/144/576, holding the old1,023-dimensional basis,1,024-context training metric and reconstruction mean fixed. All48 maps are saved before evaluation. Each half is centered on its own covariance mean; reconstruction uses the common mean.

These are separate covariance fits conditional on a shared basis and metric, not independent end-to-end discoveries. The shared basis itself used1,024 of these pooled sources, so the halves are not independent of all upstream representation choices. The four partitions reuse the same source pool and are not independent replications. H512/H521 evaluation panels are reused; no fresh superiority or equivalence claims follow.

Write-space similarity is the mean squared cosine of principal angles between orthonormal bases for each map's range. Operator difference is the Frobenius norm of A−B divided by the root mean square of the two operator Frobenius norms. The latter measures both reading and writing for oblique Fisher maps; individual eigenvector alignment is not used.

| Method | Rank | Write-space overlap range | Median angle range (degrees) | Relative operator difference range |
|---|---:|---:|---:|---:|
|PCA|36|0.830–0.835|15.059–16.394|0.574–0.582|
|PCA|144|0.715–0.718|22.079–22.602|0.750–0.755|
|PCA|576|0.833–0.835|8.270–8.392|0.575–0.577|
|Fisher|36|0.827–0.830|15.810–16.239|0.560–0.573|
|Fisher|144|0.708–0.712|23.024–23.231|0.768–0.774|
|Fisher|576|0.833–0.834|8.398–8.557|0.607–0.611|

Higher-rank subspaces have higher overlap partly by dimension alone. For independent uniformly random subspaces of equal rank in1,023 dimensions the expected overlap is k/1,023; this is a geometric reference, not an assumed null distribution for the fitted maps. Similarity should not be compared across ranks without that qualification.

Symmetric prediction KL is half KL(prediction A||B) plus half KL(B||A). It measures how differently the two fits predict, whereas approximation error is their average KL from the native model. Their ratio is descriptive, not an additive error decomposition or an equivalence test.

| Panel | Method | Rank | Recovery range across8 fits | Pair symmetric KL range | Disagreement / average error range |
|---|---|---:|---:|---:|---:|
|H512|PCA|36|69.54%–70.03%|0.0003205–0.0003573|8.91%–9.86%|
|H512|PCA|144|86.74%–87.06%|0.0005061–0.0005162|32.15%–33.08%|
|H512|PCA|576|98.15%–98.29%|0.0002109–0.0002146|100.34%–101.76%|
|H512|Fisher|36|71.22%–71.67%|0.0003006–0.0003283|8.83%–9.62%|
|H512|Fisher|144|87.98%–88.26%|0.0004643–0.0004904|32.81%–34.52%|
|H512|Fisher|576|98.60%–98.69%|0.0001571–0.0001654|97.77%–102.63%|
|H521|PCA|36|71.19%–71.48%|0.0003515–0.0003875|9.52%–10.48%|
|H521|PCA|144|87.01%–87.35%|0.0005304–0.0005693|32.18%–34.34%|
|H521|PCA|576|98.19%–98.31%|0.0002273–0.0002353|100.08%–101.69%|
|H521|Fisher|36|72.42%–72.85%|0.0003159–0.0003393|9.00%–9.61%|
|H521|Fisher|144|88.16%–88.47%|0.0005044–0.0005438|33.50%–36.17%|
|H521|Fisher|576|98.56%–98.63%|0.0001704–0.0001855|94.83%–101.50%|

Within each panel and rank, all eight fits have similar native fidelity, but prediction disagreement is substantial relative to the remaining error. Across both methods/panels, rank36 pair disagreement is about9-10% of average approximation error; rank144 is about32-36%; rank576 is about95-103%. Rank576 absolute symmetric KL is only about0.00016-0.00024. The near100% ratio concerns a small residual, not all model behavior.

For Fisher, write-space overlap is about0.83 at36,0.71 at144,0.83 at576; relative operator differences remain about0.56-0.57,0.77,0.61. Independent random576-dimensional subspaces in1,023 dimensions already have expected overlap0.563. Some principal angles remain near90 degrees. Agreement is not merely an arbitrary rotation of an identical retained subspace: both maps and predictions differ.

Compared with earlier H514512-source rank36 fits (overlap about0.66 and disagreement about18% of approximation error), current1,536-source fits show higher overlap and less relative disagreement. That historical comparison also changes shared-basis constraints and, for Fisher, the metric, so it is suggestive rather than a controlled sample-size effect. Current PCA/Fisher stability is broadly similar despite Fisher's higher fidelity.

Reproducible approximation quality does not identify a unique computation. Different fitted maps retain different weak corrections. Near-perfect fidelity to the large common effect can coexist with considerable disagreement relative to the small residual. No unique axes, semantic feature identity, or population equivalence has been established. Four overlapping partitions are not four independent confirmations.


Audits: every partition contains each calibration index exactly once across its two halves; inputs hash checked; all half covariances positive definite; covariance root/inverse and map idempotence checks pass; all per-document KL values finite. All maps, split indices, geometry summaries, per-document native approximation KL and symmetric pair KL are archived. No new documents; tracked total24,064. No CE objective or task-loss conclusion is introduced here.

Next H524: decompose paired errors into common and differential components for frozen H523 maps, without refitting. For every pair/method/rank and both reused panels, define e_common=(e_left+e_right)/2 and e_diff=(e_left-e_right)/2. Under each actual receiver's full-path local Fisher, compute their quadratic energies and cross term, replay individual local errors, and verify average error = common energy + differential energy. Evaluate finite prediction KL of midpoint reconstruction and compare with both half fits and H522 pooled3,072-source rank-k fit. The midpoint can have rank up to min(2k,1023), so it is not a matched-rank competitor or free compression; report that distinction. Local theory predicts midpoint improvement of one quarter of pairwise local KL; check against finite behavior. This distinguishes fit-dependent residual corrections from error shared across fits and tests pooling data at fixed rank versus averaging maps. Reused-panel diagnostic, not uniqueness or semantic-core evidence.

