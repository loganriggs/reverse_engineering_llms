## H513 — Similar fidelity does not identify a stable joint feature subspace

Make four prescribed random complementary218/218 splits of the original436 source-calibration documents (seed20271230). Fit joint PCA36 independently in each half, using that half's own covariance mean. For reconstruction, retain the original full-calibration mean for every fit, so this diagnostic isolates basis variation rather than intercept variation. Each split's halves are disjoint; the four splits overlap and are not four independent datasets. The shared reconstruction mean also means this is not total-estimator uncertainty.

Evaluate all eight half-fit maps on the same H512 natural panel, replacing the entire attention write before the final MLP and residual addition. Keep the original436-document PCA36 map as reference. No split or map is selected by evaluation performance. Measure principal-angle subspace agreement and symmetric KL between each pair's actual output distributions, in addition to error against the original model.

| Split | Left recovery | Right recovery | Mean squared principal cosine | Median / maximum principal angle | Pair symmetric KL | Pair disagreement / average half-map error |
|---|---:|---:|---:|---|---:|---:|
|0|63.37%|64.10%|0.530|39.18° / 88.23°|0.001066|24.68%|
|1|63.58%|64.07%|0.509|39.38° / 89.25°|0.001030|23.90%|
|2|64.08%|63.99%|0.524|41.27° / 89.80°|0.001042|24.33%|
|3|63.40%|63.56%|0.525|38.06° / 89.36°|0.001031|23.70%|

All eight half fits have similar average fidelity: mean KL0.004277–0.004363 and recovery63.37–64.10% relative to mean-source replacement. The full436-document fit has lower KL0.003897 (67.29% recovery on this panel). The difference is consistent with a calibration sample-size effect, but one full fit versus overlapping half fits is not a controlled variance-reduction estimate.

The subspaces are much less consistent than their scalar performance: mean squared principal cosine is0.509–0.530, median angles38.06–41.27°, and maximum angles88.23–89.80°. Principal angles are invariant to rotations of axes within each subspace, so these differences cannot be removed just by relabeling, signing, or rotating the36 coordinates. This experiment does not separately identify stable individual axes.

Paired output disagreement is0.001030–0.001066 symmetric KL, about8.64–8.95% of the mean-replacement effect and23.70–24.68% of the average half-map error. These are distinct distributions despite similar overall fidelity. The error ratios use different KL reference directions and are descriptive scales, not additive error decompositions or equivalence tests.

The present evidence supports a useful low-dimensional approximation whose exact subspace remains sample-sensitive. It does not support calling these particular36 directions canonical or fundamental. The uncertainty may concentrate in weak covariance directions, reflect limited calibration size, or correspond to genuinely different near-equal-quality approximations. More calibration data is needed to distinguish those possibilities; unstable finite-sample fits do not prove the population representation is nonunique.

Audits: each pair partitions the436 calibration IDs into two disjoint218-document halves; all split IDs and eight fitted bases/spectra are archived. Original full-fit per-case KL replays H512 within tolerance; all rows are finite and aggregate results replay. Shared global mean and half-specific covariance means are explicit. No documents were added; tracked total remains20,480. All outcome comparisons use a reused evaluation panel and have no fresh confirmation intervals.

Next H514: collect1,024 new complete-document-disjoint calibration documents, split them into two prescribed independent512-document pools, and fit joint PCA36 in each pool and in their pooled1,024 sample. Keep the original reconstruction mean fixed across these new fits and the earlier436-document reference, while centering each fitting pool around its own mean for covariance estimation. Evaluate all maps on the existing H512 panel without selecting the best fit. Compare the two512-sample subspaces and predictions, and report pooled1,024 fidelity relative to both. This tests whether substantially more calibration improves the representation and gives a less sample-limited stability check. One pooled fit still does not establish asymptotic uniqueness or a variance estimate. Preserve document exclusions against all20,480 tracked documents; these new documents are calibration data, not fresh evaluation evidence.
