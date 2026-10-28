## H723 — Choosing the write metric reveals much more compact agreement effects

Previous goal turn classification: progress. H722 showed task-blind head-coordinate PCA can preserve causal effects despite substantial reconstruction error, especially for agreement. H723 changes only the reconstruction metric before introducing a more flexible architecture.

For each head, factor its fixed output projection W=U diag(s) V^T. Map a row of128head coordinates h into y=h V diag(s). Euclidean distance in y is exactly the residual-write distance because Uhas orthonormal columns. Fit centered PCA to y using only the same32768natural-text training vectors; map reconstructions back through diag(1/s)V^T. This is rank-constrained linear PCA optimized for the write the head sends to the residual stream. It is not PCA optimized for output KL or task answers. Site selection remains task-informed; representation fitting remains task-blind.

Reuse the same natural validation split, task/nuisance panel and32document output-fidelity panel asH722. Project task donor-minus-receiver head differences, while natural-output fidelity replaces the entire head activation by its reconstruction. Keep downstream computation live and original corpus reservedtest untouched.

| Head | Rank | Coordinate-PCA fraction of full-head transfer effect | Write-PCA fraction | Write-PCA transfer fraction95%interval |
|---|---:|---:|---:|---|
|agreement|1|15.7%|60.0%|[58.9%,61.3%]|
|agreement|4|68.9%|90.5%|[89.7%,91.4%]|
|agreement|8|79.1%|96.3%|[95.4%,97.2%]|
|agreement|32|94.9%|99.2%|[99.0%,99.4%]|
|agreement|64|97.1%|99.9%|[99.8%,100.1%]|
|capital_recall|1|0.2%|0.5%|[0.1%,0.9%]|
|capital_recall|4|3.4%|10.2%|[4.6%,15.6%]|
|capital_recall|8|10.4%|14.3%|[9.1%,18.2%]|
|capital_recall|32|46.7%|54.5%|[46.7%,61.3%]|
|capital_recall|64|74.3%|83.5%|[77.3%,88.5%]|

Agreement's task effect is much more concentrated under the write metric: one direction retains60.0%, four retain90.5%, eight96.3%,and32retain99.2%of this head's transfer-margin effect. The corresponding coordinate-PCA values are15.7%,68.9%,79.1%,94.9%. These fractions concern an already partial head intervention, not task accuracy or the full model algorithm. Pair-bootstrap intervals resample8transfer pairs with both task directions together and recompute the ratio of mean gains;10000resamples. They condition on the selected head, fitted representation and reused task suite, and do not establish universal agreement dimensions.

Capital recall improves more modestly and remains distributed under this fixed linear basis:32dimensions retain54.5%of the head effect,64retain83.5%,96retain96.7%. This is evidence against assuming that one compact fixed subspace will suffice equally for every computational target. It does not rule out lower-dimensional per-example supports, a different metric or a nonlinear/shared-group representation.

### Reconstruction and output fidelity remain separate objectives

Write-PCA reduces validation writeMSE at every tested nontrivial rank. At rank32, agreement relative writeMSE falls0.2743→0.2314andcapital0.4730→0.4444. But this improvement does not guarantee improved downstream KL. At rank8, mean native-to-reconstruction KL rises0.000993→0.001193foragreement and0.007315→0.007703forcapital. Paired32document bootstrap intervals for both increases include zero, so do not call this a reliable deterioration. It is nonetheless not evidence of uniform downstream improvement.

At rank32, meanKLfalls0.000428→0.000331agreement and0.002393→0.001793capital. The agreement difference has a paired bootstrap interval below zero; the capital difference does not. Rank64also lowers the means, with the same distinction in interval support. These exploratory rank comparisons are not multiplicity-adjusted. Observed continuationCE remains separately recorded; neither KL nor CE is answer correctness.

Thus internal coordinate reconstruction, residual-write reconstruction, native-output faithfulness and preservation of a particular task effect can rank representations differently. This supports the user's proposed spectrum of data/function-based equivalences, while demonstrating why the equivalence criterion must be stated explicitly. A compact agreement-preserving span is not yet a canonical unit of computation or a meaningful manifold.

Next move beyond fixed linear subspaces: train multiple-seed full-span whitened sparse rotations and a matched signed-BSF baseline on these natural head activations. Use the established write-space metric, track active dimensions/parameters and fit stability, then repeat the frozen task/nuisance and native-output tests. Only after stable reusable directions or subspaces emerge should overlapping memberships/hierarchical gates be interpreted. The prior synthetic architecture tests remain useful, but real-data comparisons now have a stronger causal target and a demanding task-blind linear baseline.

Audits:output-projection condition numbers4.21agreement and2.31capital, fullrank encoder-decoder inverse errors≤1.34e-15, write-isometry errors≤2.48e-15. Independent direct training reconstruction matches the PCA eigenvalue floor within2.41e-16relative to total write energy. Fullrank task swaps replay within3.82e-6margin error; fullrank native reconstruction maxKL3.65e-12.1152whole-model evaluations, no new capture passes. All source/model/data hashes unchanged, finite metrics, original reservedtest untouched, no new unique research documents; corpus pool26112. Broad goal active.
