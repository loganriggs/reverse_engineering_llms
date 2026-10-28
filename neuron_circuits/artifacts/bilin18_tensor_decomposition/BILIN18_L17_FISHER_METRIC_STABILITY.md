## H503 — Stability of probability-aware source maps across metric samples

Fit four rank-4 Fisher source maps for each of nine heads, using four prescribed disjoint groups of 16 training receivers. Group offsets are 0, 16, 32, and 48 with stride64 through the first1,024 H478 training receivers. Group0 reproduces the original H498 fit. Keep the source calibration basis, covariance, rank, and construction fixed. Evaluate all four fits on the same H500 donor/H499 receiver pairs, both intervention signs, with native downstream computation. There is no best-fit selection. This is a diagnostic on reused evaluation data, not fresh confirmation.

Compare three distinct notions of stability: the four-dimensional write subspace; the full oblique read/write operator, accounting for calibration source covariance; and the native output distributions induced by the reconstructed sources. Similar aggregate recovery alone does not prove similar predictions.

Write overlap below is the mean squared cosine of the four principal angles between two write subspaces (1 means the same subspace). Prediction disagreement is symmetric KL between two reconstructed-source output distributions, divided by the full native intervention effect KL. Ranges cover all six fit pairs, with both signs included in the prediction-disagreement range. Recovery ranges cover the four fits separately for each sign.

| Head | Recovery range negative | Recovery range positive | Write-subspace overlap range | Pair disagreement / native effect |
|---|---|---|---|---|
|0|18.13%–22.20%|18.10%–27.56%|0.678–0.836|9.36%–16.35%|
|1|81.71%–82.78%|80.21%–81.34%|0.947–0.980|1.30%–4.71%|
|2|48.54%–56.91%|53.30%–62.02%|0.555–0.932|6.04%–26.78%|
|3|64.00%–68.23%|59.58%–64.94%|0.629–0.871|5.86%–10.66%|
|4|80.83%–82.54%|81.26%–82.50%|0.701–0.775|1.41%–4.40%|
|5|55.30%–58.22%|54.06%–57.52%|0.713–0.792|4.71%–10.20%|
|6|98.55%–98.72%|98.30%–98.50%|0.742–0.945|0.15%–0.42%|
|7|51.03%–55.44%|51.69%–55.63%|0.683–0.851|4.45%–6.88%|
|8|81.78%–82.73%|80.99%–82.38%|0.727–0.955|2.10%–5.67%|

Head6 is behaviorally stable at this resolution: recovery is98.30–98.72% across both signs/fits, while pair disagreement is only0.15–0.42% of native effect. Its write-subspace overlap nevertheless ranges0.742–0.945. Some geometric variation therefore concerns directions with little behavioral influence under this donor/receiver distribution.

Head2 is more sensitive to the metric sample: its negative-sign recovery ranges48.54–56.91%, positive53.30–62.02%, and pair disagreement reaches26.78% of native effect. Write overlap ranges0.555–0.932. Head0 is also poorly captured and unstable, with positive-sign recovery varying18.10–27.56% and pair disagreement9.36–16.35%. This supports metric-estimation sensitivity as a relevant limitation in some heads, but does not prove that a larger sample will fix it.

Head5's large advantage over PCA is qualitatively reproducible across these four fits: recovery remains about54–58%, compared with H500 PCA20.51–22.19%. However, pair disagreement is4.71–10.20% of native effect. This is not exact functional equivalence. Heads1,4,8 have relatively narrow recovery ranges but nonzero pairwise output differences. Similar scalar performance should not be mistaken for the same computation.

The full operator comparison is invariant to rotations of internal four-dimensional coordinates. Define A in the fixed128-dimensional calibration PCA basis and C as its diagonal covariance. Report ||(Ai−Aj) C^(1/2)||F divided by sqrt((||Ai C^(1/2)||F²+||Aj C^(1/2)||F²)/2). This measures variation in reconstructed source vectors under calibration covariance, not variation in output probability:

| Head | Covariance-weighted operator relative difference range |
|---|---|
|0|0.696–0.851|
|1|0.098–0.153|
|2|0.414–0.847|
|3|0.530–0.730|
|4|0.281–0.335|
|5|0.570–0.682|
|6|0.035–0.047|
|7|0.520–0.688|
|8|0.382–0.667|

This experiment does not measure individual-axis stability separately from subspace/operator stability; no unique axes are established. It also holds source covariance fixed, so it excludes calibration-source estimation uncertainty. The four training groups are disjoint prescribed subsets of the same training pool, not four independent training corpora. There are no confidence intervals or fresh-source/receiver claims here.

Audits: all64 metric-training IDs are distinct; group0 reproduces the archived original metric and the original finite responses (largest relative replay discrepancy below1e-14). All per-case errors and six pairwise symmetric-KL arrays are finite; aggregate recoveries replay. Fitted Fisher matrices, rank-4 eigenspaces, and complete128-dimensional operators are archived for every head/group. The geometric formulas remove internal basis-rotation ambiguity. No new documents were consumed; tracked total remains19,456.

The supported picture is heterogeneous: downstream-metric-aware low-rank operators can retain useful, reproducible behavior, but neither a four-axis geometry nor a scalar fidelity score gives a universal canonical unit. Head6 is close to a stable behavioral approximation; heads0/2 reveal substantial metric-sample sensitivity; head5 retains a strong benefit without exact functional identity across fits.

Next H504: form the pooled64-receiver Fisher metric by averaging the four equal-sized fitted metric matrices, then recompute the prescribed rank-4 map for every head. Compare against the original16-receiver map and report all four individual-fit recoveries for context, on the same fixed evaluation pairs. Do not average already-truncated operators: truncation and metric averaging are different operations. This cheaply tests whether a larger metric sample improves the unstable cases without changing rank or source covariance. Treat it as diagnostic and do not select the best individual fit by evaluation. Fresh confirmation should be reserved for a specific supported claim. Calibration-source uncertainty and observed-token CE remain open.
