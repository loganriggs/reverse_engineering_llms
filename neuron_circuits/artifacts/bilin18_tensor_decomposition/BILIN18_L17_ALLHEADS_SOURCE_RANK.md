## H497 — Source-rank requirements differ sharply across heads

Follow H496 by varying the source projection rank while retaining native downstream MLP computation. The prescribed grid is 4, 16, 64, and 128 for all nine heads. Each basis is the leading principal components of that head's centered calibration writes, using the same calibration IDs as H496. Evaluate the same 64 held-out donors, each assigned to eight of the same 512 receiver contexts, in both intervention signs. There is no cross-map compression in this experiment. This isolates whether the source projection itself preserves the native finite intervention response.

Recovery is 1 minus pooled KL(full native target || projected-source native target) divided by pooled KL(full native target || intact baseline). It measures preservation of a specific intervention's output-distribution effect, not token accuracy. All evaluation receivers were used previously; these are exploratory point estimates, not fresh confirmation or equivalence tests.

| Head | Rank 4 recovery −/+ | Rank 16 recovery −/+ | Rank 64 recovery −/+ | Rank 128 recovery −/+ |
|---|---|---|---|---|
|0|31.16% / 31.00%|56.64% / 59.24%|87.67% / 87.73%|100.00% / 100.00%|
|1|84.68% / 83.94%|90.12% / 89.47%|97.66% / 97.48%|100.00% / 100.00%|
|2|33.46% / 32.03%|61.72% / 63.65%|94.37% / 94.88%|100.00% / 100.00%|
|3|32.52% / 30.36%|50.87% / 48.76%|85.79% / 85.50%|100.00% / 100.00%|
|4|83.99% / 82.93%|88.57% / 87.26%|95.98% / 95.89%|100.00% / 100.00%|
|5|9.59% / 8.55%|61.66% / 62.26%|87.02% / 87.21%|100.00% / 100.00%|
|6|98.90% / 98.73%|99.15% / 99.20%|99.05% / 99.56%|100.00% / 100.00%|
|7|43.48% / 41.55%|53.18% / 51.84%|82.15% / 81.27%|100.00% / 100.00%|
|8|76.32% / 78.62%|84.44% / 85.85%|95.83% / 96.03%|100.00% / 100.00%|

Head 6 is already at 98.73–98.90% recovery with four source coordinates. At rank 16, the other heads range from 48.76% to 90.12%; at rank 64 they range from 81.27% to 97.66%. Rank 128 recovers all heads numerically, as expected when the calibrated span contains the full head output space. This endpoint is a reconstruction audit, not evidence of useful compression. The successful endpoint also makes an unexplained source-span mismatch unlikely in this panel.

All non-head-6 cases improve across the tested ranks. Head 6's negative-sign recovery decreases slightly from 99.15% at rank 16 to 99.05% at rank 64, while its positive-sign recovery increases. Although nested PCA projections monotonically reduce raw source squared error, they need not monotonically reduce finite downstream KL. We should not impose monotonic behavioral recovery as an audit or infer a reliable decline from this exploratory difference.

| Head | Donor source energy retained at rank 4 | Rank 16 | Rank 64 |
|---|---:|---:|---:|
|0|22.59%|50.92%|86.29%|
|1|82.23%|88.12%|96.14%|
|2|36.07%|65.81%|92.28%|
|3|19.28%|37.16%|78.01%|
|4|51.70%|62.43%|86.44%|
|5|10.15%|33.20%|75.35%|
|6|97.94%|98.44%|99.39%|
|7|23.54%|40.82%|77.16%|
|8|30.37%|54.42%|84.76%|

The metric mismatch is visible in several heads. Head 4's rank-4 representation retains 51.70% of source energy but preserves roughly 83–84% of effect KL. Head 8 retains 30.37% of source energy but preserves 76–79% of effect KL. Head 5 improves from about 9% to 62% recovery between ranks 4 and 16, while source energy rises from about 10% to 33%. These observations motivate selecting source coordinates by downstream effect, not solely by raw activation variance. They do not yet demonstrate that a particular alternative basis will outperform PCA.

Combined with H496, the evidence separates two limitations: many heads require more source information than four PCA coordinates, and some heads also incur substantial additional error from the rank-64 cross-map approximation. The losses are not additive KL components. This experiment does not train a more accurate compiled program, remove its native baseline dependency, establish semantic features, or measure observed-token CE.

Audits: all 18 rank-4 head/sign aggregates replay H496 with zero recovery drift; every rank-128 source reconstruction has relative error below 1e-5 and effect-KL recovery within 1e-4 of one; all per-case arrays are finite and aggregate metrics replay. The final JSON contains exact source reconstruction errors and all metrics; per-case full effect KL, prediction KL, raw effect energy, and raw prediction error are archived. No new documents were consumed; the tracked total remains 18,880.

Next discrimination: compare PCA source projections with a downstream-metric-aware low-rank source map at matched ranks. Estimate the average local output Fisher metric over a fixed subset of training receivers, combine it with calibration source covariance, and select source directions using that weighted metric. Freeze training receiver IDs, rank grid, and construction before evaluating finite signed interventions. Include all heads, report raw source distortion as well as KL, and retain the native downstream computation to avoid mixing source selection with cross-map compression. A local Fisher objective is only a hypothesis about finite-KL fidelity; improvement, failure from finite-intervention nonlinearity, and unstable oblique projection are all plausible. A rank-128 reconstruction endpoint and derivative checks are necessary implementation audits. This prepares a concrete comparison of raw-data and behavioral notions of canonical form. Full write/intervention/KL/observed-token-CE objective comparisons remain open.
