## H460 — Own-head versus shared source spaces

Reuse H459108 source writes and128 receiving contexts, at−1/+1. Exclude all12 evaluation donor IDs from the512-document donor calibration pool, leaving500 baseline donor documents. Fit uncentered source SVD separately per head at ranks16/32/64. Fit a shared covariance by averaging per-head uncentered covariances after normalizing each to unit trace; use shared ranks16/144. Freeze before response evaluation; no finite outcomes enter fitting. Earlier frozen model/source artifacts still have their historical calibration scope; this disjoint split does not make the whole pipeline independent. Total12992 documents.

Evaluate the exact native omitted response to each reconstructed source, plus the exact retained response to the original source. There is no context-input or output compression in this diagnostic, so poor transfer cannot be blamed on the head6 input encoder. Keep the head6-specific retained128 factors fixed. The original frozen head6 reconstruction is a replay control.

| Head | Own16 | Own32 | Own64 | Shared16 | Shared144 | Frozen head6 | Own16 source error |
|---|---:|---:|---:|---:|---:|---:|---:|
|0|67.07%|78.83%|90.68%|40.68%|76.69%|24.05%|0.4731|
|1|96.89%|97.90%|98.98%|93.32%|94.89%|50.98%|0.0381|
|2|87.98%|92.20%|97.01%|71.59%|90.92%|36.98%|0.2551|
|3|75.35%|81.52%|89.99%|60.61%|82.24%|49.18%|0.5940|
|4|79.27%|86.46%|93.16%|70.03%|80.61%|32.96%|0.4642|
|5|59.22%|69.50%|82.07%|41.52%|67.27%|29.92%|0.5459|
|6|99.48%|99.59%|99.80%|99.03%|99.24%|99.54%|0.0095|
|7|66.79%|75.66%|87.58%|58.73%|75.67%|42.33%|0.6244|
|8|92.68%|95.64%|97.95%|86.60%|94.66%|78.07%|0.3998|

Table shows positive-amplitude finite-KL recovery; full signed/raw/Fisher arrays are saved. Source error is pooled squared reconstruction error divided by source energy. Uncentered SVD includes source means; high coverage is not automatically evidence for varying semantic features.

| Frozen contrast | Estimate | Joint95% interval | Required lower | Pass |
|---|---:|---|---:|---|
|own16_nonhead6_recovery_negative|+0.781616|[+0.739192, +0.824040]|+0.90|False|
|own16_nonhead6_recovery_positive|+0.781572|[+0.739148, +0.823996]|+0.90|False|
|own16_minus_shared16_negative|+0.127193|[+0.084769, +0.169616]|+0.05|True|
|own16_minus_shared16_positive|+0.127715|[+0.085291, +0.170139]|+0.05|True|

Primary own16 recovery is the unweighted mean across eight non-head6 heads, threshold90%, at each sign; own16-minus-shared16 advantage threshold5pp at each sign.4000 bootstrap replicates pair donors across heads and resample receiving contexts; heads are the fixed target population. Joint95% max-absolute-deviation bands cover four contrasts. Source calibration uncertainty is not resampled. Higher ranks and shared144 are descriptive controls, not selected replacements for failed gates.

Audits: calibration/evaluation donor IDs disjoint; native frozen-head6 reference replay max absolute error3.28e-07, native effect replay5.5e-07, both below1e-6. All arrays finite, artifact hashes checked and no head-specific exclusions.

Own-head rank16 spaces improve strongly over the shared rank16 control, but they do not provide uniformly high response coverage. Non-head6 mean own16 recovery is78.162% at−1 and78.157% at+1, joint95% intervals[73.919,82.404]% and[73.915,82.400]%. Both frozen90% coverage gates fail. Own16-minus-shared16 gains12.719pp/12.772pp, intervals[8.477,16.962] and[8.529,17.014]pp, pass both5pp advantage gates. These tests establish the value of source identity at this budget, not a universal16-dimensional unit.

Positive own16 recovery varies sharply: head6 99.482%, head1 96.888%, head8 92.683%, head2 87.985%, head4 79.268%, head3 75.351%, head0 67.071%, head7 66.793%, head5 59.219%. Rank64 improves all heads on point estimates, but head5 remains82.073%, head7 87.581% and head3 89.989%; these ranks are descriptive, not selected successes. Head6 source reconstruction error at own16 is0.950%, while heads0/3/4/5/7 lose47–62% of raw source energy. Head6 is therefore unusually low-dimensional in this natural-source experiment; it should not anchor a universal rank claim.

Shared144 positive recovery remains below own16 for some already compact heads (head1/head6) and above own16 for several broader ones, consistent with allocation tradeoffs. It does not uniformly dominate despite using144 coordinates per intervention. Own16 and shared144 have equal source-basis storage only; their computational programs and routing information differ. No causal preference for separate semantic modules follows from that budget comparison alone.

The leading picture is heterogeneous source-conditioned computation: head-specific coordinates matter, but source dimensionality varies greatly. The earlier frozen operation’s cross-head failure was not fixed simply by swapping in equally small own-head spaces. Yet uncentered source SVD includes mean activity; high head6/head1 coverage may reflect a persistent source offset rather than a rich low-rank family of varying features. Natural donor experiments also do not exhaust task-specific or counterfactual source directions.

Next explicitly separate source means from variation. Calibrate per-head means and centered covariance on the same disjoint donor split; compare mean-only, uncentered rank16 and mean-plus-centered-rank16 reconstructions on full donor interventions. Also intervene with centered donor deviations and test their centered-rank16 reconstruction, reporting absolute effect magnitudes as well as recovery. This discriminates a stable mean-driven late interface from genuinely compact context-varying computations before further program compilation or semantic naming. Preserve current coverage failures and source-specific budget accounting; independent confirmation is still required for any newly selected explanation.

Budget/scope: nine own16 bases store9*16*1152=165888 coefficients; shared144 stores the same source-basis count but reads144 rather than16 coordinates per source. Its quadratic source-pair program would require10440 writes versus9*136=1224 for nine own16 programs. Own16 versus shared16 matches per-intervention coordinate count but not total storage or supplied head identity. This experiment evaluates native omitted computation and does not yet compile/compress a family of head-specific operations.
