## H521 — Fresh confirmation of larger training metrics

Freeze H520's1,024-context Fisher maps at ranks36/144/576 and compare them with the64-context Fisher maps and PCA at each rank on512 fresh documents. All maps share the same source basis, covariance and reconstruction mean. The configuration, ranks, comparisons and bootstrap rule were fixed before evaluation; no model or rank is selected on this panel.

| Rank | PCA prediction KL | Fisher64 KL | Fisher1024 KL | KL reduction vs64 | KL reduction vs PCA | Fisher1024 recovery |
|---|---:|---:|---:|---:|---:|---:|
|36|0.003762451|0.003735130|0.003595400|3.74%|4.44%|72.11%|
|144|0.001729058|0.001727892|0.001605549|7.08%|7.14%|87.55%|
|576|0.000278303|0.000245329|0.000219649|10.47%|21.08%|98.30%|

Recovery is relative to replacing the attention contribution by its frozen mean, whose prediction KL is0.012893 on this panel. It is not model accuracy or total computation explained. KL is original-model to replacement prediction divergence under a full residual-path replacement, with native downstream computation retained.

Six paired mean-KL gains were evaluated together using4,000 document bootstrap resamples (seed20280104), centered max-absolute standardized deviations with fixed original paired standard errors, and a simultaneous95% two-sided interval family. A positive lower bound supports the corresponding superiority claim; a failed bound is not evidence of equivalence.

| Comparison | Mean KL reduction | Simultaneous lower | Simultaneous upper | Positive lower bound |
|---|---:|---:|---:|---|
|n1024_rank36_over_n64_rank36|0.000139731|0.000027039|0.000252423|True|
|n1024_rank36_over_PCA36|0.000167051|-0.000010636|0.000344739|False|
|n1024_rank144_over_n64_rank144|0.000122344|0.000065870|0.000178817|True|
|n1024_rank144_over_PCA144|0.000123509|0.000042083|0.000204936|True|
|n1024_rank576_over_n64_rank576|0.000025680|0.000015287|0.000036073|True|
|n1024_rank576_over_PCA576|0.000058654|0.000045307|0.000072001|True|

All three Fisher1024-versus-Fisher64 comparisons pass. Prediction error decreases by3.74%,7.08%,10.47% at ranks36,144,576. This independently confirms that the training receiver sample matters to the learned behavioral coordinates. The experiment changes context sample count and composition together; it does not isolate a pure sample-size effect or establish a universal convergence law.

Against PCA, Fisher1024 passes at144 and576, with7.14% and21.08% relative error reductions. The rank36 point reduction is4.44%, but its simultaneous interval includes zero. Behavioral weighting is useful at the higher tested ranks; a reliable advantage at the tightest tested rank remains unestablished. The broad rank spectrum persists: Fisher1024 recovery is about72.11%,87.55%,98.30%.

Observed-token CE changes for Fisher1024 are+0.001344±0.003685,−0.000383±0.002370, and−0.001477±0.000932 (mean±SE). These are descriptive; no loss-neutrality, task improvement, or CE-optimality claim follows. The different rank36 CE/KL point orderings also cannot establish an objective tradeoff without a dedicated comparison.

Data audit: independent rehashing verifies that all512 complete documents and their129-token windows match their source cache, and none overlaps the21,504 previously tracked complete documents, including H514 calibration. Tracked total is22,016. This is fresh relative to the continuation's tracked panels, not a model-training holdout guarantee. All frozen inputs are hash checked; head-sum reconstruction passes; stored per-document errors are finite; independent bootstrap replay from the archived errors reproduces every simultaneous bound. The original64-context endpoints were replayed in H520. No standalone MLP replacement or deployment savings are claimed.

Updated inference: metric estimation is now a confirmed contributor to the performance of these data-based coordinates. Source-moment estimation remains a separate leading limitation (H518). Neither result establishes unique axes or canonical equivalence outside the sampled distribution.

Next H522: isolate source-calibration sample sensitivity while holding the1,024-context training metric, old1,023-dimensional supported basis, reconstruction mean and ranks fixed. Collect2,048 new source-calibration documents, auditing all prior exclusions including H521. Form nested source pools of1,024 (existing),2,048 (existing plus first1,024 new), and3,072 (existing plus all new). Fit each covariance around its own calibration mean in the fixed basis, while reconstruction still uses the shared frozen mean. Use the full covariance matrix for the larger pools; compute symmetric square roots/inverses and verify the original endpoint, avoiding an incorrect diagonal-covariance assumption in the old PCA basis. Fit both covariance-PCA and Fisher maps for each pool, save before evaluation, and report all outcomes on the already-used H512/H521 panels as diagnostics. New confirmation is required for subsequent superiority claims. This separates source estimation from receiver-metric estimation without expanding the supported basis or adding context-dependent machinery.
