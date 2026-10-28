## H500 — Probability-aware coordinates on fresh donor sources

Test the frozen H498 rank-4 Fisher and PCA source maps on 64 new donor documents. Complete-document hashes are disjoint from all 19,392 tracked prior documents, including H499 receivers. Keep the 512 H499 receivers, all nine heads, both signed unit interventions, original calibration means, and fitted maps fixed. New source writes are centered by the original calibration mean; they are not recentered, individually normalized, or refitted on the new pool. Each donor is paired with eight receivers. Native downstream computation is retained.

This checks source generalization separately from H499's receiver-generalization check. Because all maps remain frozen, no new source outcome tunes the representation. Receivers are reused, so this is not a new independent receiver confirmation. The new donors and the H499 receivers were both excluded from original coordinate fitting.

Report gain as Fisher minus PCA KL recovery. The 95% intervals are simultaneous across all 18 head/sign comparisons. Unlike H499's receiver-within-donor bootstrap, H500 resamples 64 donor groups with their eight associated receivers intact, paired across methods/heads/signs. This reflects donor-group variation while conditioning on the frozen maps; it does not independently vary source and receiver sampling. The two studies' pass counts are not directly comparable measures of robustness.

| Head | Sign | PCA recovery | Fisher recovery | Gain (pp) | Simultaneous 95% gain interval (pp) |
|---|---|---:|---:|---:|---|
|0|negative|24.01%|21.11%|-2.90|[-11.07, +5.27]|
|0|positive|26.61%|27.56%|+0.94|[-8.34, +10.23]|
|1|negative|79.47%|82.78%|+3.31|[+0.29, +6.32]|
|1|positive|78.23%|81.19%|+2.96|[-0.20, +6.12]|
|2|negative|37.38%|48.54%|+11.16|[-5.68, +28.01]|
|2|positive|41.11%|53.30%|+12.19|[-4.70, +29.08]|
|3|negative|35.02%|68.23%|+33.21|[+1.11, +65.31]|
|3|positive|33.48%|64.94%|+31.45|[+3.43, +59.47]|
|4|negative|83.44%|82.54%|-0.90|[-3.64, +1.84]|
|4|positive|82.61%|82.50%|-0.11|[-2.47, +2.26]|
|5|negative|22.19%|57.79%|+35.60|[+7.60, +63.59]|
|5|positive|20.51%|57.52%|+37.00|[+9.19, +64.81]|
|6|negative|98.04%|98.69%|+0.65|[-0.31, +1.62]|
|6|positive|97.65%|98.46%|+0.81|[-0.28, +1.90]|
|7|negative|39.26%|51.03%|+11.78|[+0.35, +23.20]|
|7|positive|39.41%|51.69%|+12.29|[-0.34, +24.91]|
|8|negative|69.75%|82.62%|+12.87|[+1.55, +24.18]|
|8|positive|69.54%|82.38%|+12.84|[+1.78, +23.91]|

Eight simultaneous lower bounds exceed zero: both signs for heads 3, 5, and 8, plus the negative signs for heads 1 and 7. Head 5 remains a clear example: PCA gives 20.51–22.19% recovery and the Fisher map 57.52–57.79%, gaining 35.60–37.00 percentage points. Head 3 gains 31.45–33.21 points, but intervals are wide. Head 6 stays near 98–99% recovery under both maps, with small gains not established under donor-group uncertainty. Heads 0 and 4 remain weak or inconsistent cases. Non-passing intervals do not establish equivalence or absence of benefit.

The source pool shifts materially in raw energy. Mean centered source squared norm divided by calibration covariance trace is:

| Head | Old donor pool | New donor pool | New PCA source energy retained | New Fisher source raw reconstruction recovery |
|---|---:|---:|---:|---:|
|0|0.965|1.110|24.10%|21.26%|
|1|0.878|0.928|79.48%|77.98%|
|2|1.052|1.198|37.52%|28.20%|
|3|1.015|1.501|19.22%|14.37%|
|4|0.660|0.580|54.77%|52.54%|
|5|0.525|1.466|11.31%|7.83%|
|6|1.111|0.749|97.61%|97.47%|
|7|0.866|0.730|19.72%|15.81%|
|8|0.934|0.968|29.16%|24.90%|

For head 5, the energy ratio changes from 0.525 to 1.466, yet behavioral recovery remains around 58% for the frozen Fisher map. Its raw source reconstruction is worse than PCA (7.83% versus 11.31%) while its behavior reconstruction is much better. This is direct evidence that minimizing raw source error and preserving output probabilities favor different representations. These pool summaries do not isolate covariance changes from higher moments or source meaning, and no norm matching was performed.

The supported claim is limited but useful: some low-rank probability-aware source maps generalize to both new receivers and new donors. This does not establish unique canonical axes, semantic units, universal head transfer, full-model replacement, or arbitrary source-distribution robustness. PCA uses orthogonal projection and the Fisher map uses oblique projection; the current comparison therefore mixes selected subspace and coordinate-reconstruction rule.

Audits: independently rehashed all 19,392 excluded documents plus 64 new donors; zero overlap, all token windows verified against each manifest's source cache. Tracked total is 19,456. Frozen inputs were hash-checked. New sources lie in the full calibrated head span to relative error below 1.5e-7. All per-case results are finite and replay aggregate gains. Source arrays and pool energy diagnostics are saved. Bootstrap critical value is approximately 3.009. Document disjointness is relative to this continuation, not asserted model-pretraining disjointness.

Next H501: orthogonally project sources into the exact same four-dimensional write subspace used by each frozen Fisher map, then compare finite native responses with the existing oblique map and PCA on this same donor/receiver panel. This removes oblique read-coordinate freedom while retaining the selected output subspace. It is a diagnostic on reused data, not fresh confirmation. Report all heads/signs and separately compare orthogonal-write versus PCA and oblique versus orthogonal-write. A remaining gain for the orthogonal-write control supports behavioral subspace choice; an extra oblique gain supports the importance of how source coordinates are read. Neither alone identifies a unique feature basis.
