## H498 — Choosing source coordinates by their output effects

Compare the PCA source projections in H497 with a data-and-behavior-weighted map at the same ranks 4, 16, 64, and 128, separately for all nine heads. The behavior metric is the local output Fisher metric: the probability-weighted squared change in centered logits, which approximates twice the output KL for small perturbations. It includes the MLP, input and output RMS normalization, vocabulary readout, and logit softcap. The direct residual receiver is held fixed, matching the intervention definition.

Estimate this metric on 16 fixed training receivers (indices 0, 64, …, 960 from the earlier 1,024 training contexts), independently of the reused evaluation panel. In the full 128-dimensional calibration PCA coordinates, let C be calibration covariance and F the mean local Fisher matrix. Let U contain the leading eigenvectors of C^(1/2) F C^(1/2). The source reconstruction is A = C^(1/2) U U^T C^(-1/2). This rank-k map minimizes the calibration-covariance-averaged local quadratic reconstruction objective for that F. No finite evaluation loss, regularization strength, or clipping was used to select it.

This is generally an oblique projection: its read and write directions differ, and it can increase raw source norm. It has the same bottleneck rank as PCA, not the same orthogonal-projection constraint or necessarily the same parameter count. Its local optimality does not guarantee finite-KL optimality or generalization. Evaluate the same 64 donors and 512 receiving contexts, both signed unit interventions, with native downstream computation. These are exploratory reused-receiver results.

| Head | Rank-4 PCA KL recovery −/+ | Rank-4 Fisher map recovery −/+ | Gain (percentage points) −/+ |
|---|---|---|---|
|0|31.16% / 31.00%|31.33% / 32.50%|+0.17 / +1.50|
|1|84.68% / 83.94%|91.31% / 90.50%|+6.64 / +6.56|
|2|33.46% / 32.03%|47.55% / 47.91%|+14.09 / +15.88|
|3|32.52% / 30.36%|55.42% / 54.45%|+22.89 / +24.08|
|4|83.99% / 82.93%|86.09% / 83.77%|+2.10 / +0.84|
|5|9.59% / 8.55%|67.24% / 68.82%|+57.65 / +60.27|
|6|98.90% / 98.73%|99.04% / 99.11%|+0.15 / +0.38|
|7|43.48% / 41.55%|58.83% / 58.82%|+15.35 / +17.27|
|8|76.32% / 78.62%|83.45% / 84.60%|+7.13 / +5.99|

At rank 4, all 18 head/sign point estimates improve. Head 5 changes most, from 8.55–9.59% to 67.24–68.82% recovery. Heads 2, 3, and 7 gain roughly 14–24 percentage points; head 6 improves only slightly because its PCA projection already captures almost all behavioral effect. These gains show that low raw source variance does not imply low behavioral value. They do not yet show fresh-receiver or fresh-source confirmation, and head 5's absolute effect remains much smaller than head 6's.

| Head | Fisher rank-16 KL recovery −/+ | Fisher rank-64 KL recovery −/+ |
|---|---|---|
|0|59.08% / 59.74%|87.88% / 87.59%|
|1|94.20% / 93.49%|98.46% / 98.35%|
|2|81.46% / 83.12%|95.52% / 96.04%|
|3|63.51% / 62.83%|87.95% / 87.70%|
|4|88.41% / 85.80%|95.55% / 95.46%|
|5|74.09% / 75.03%|86.46% / 87.00%|
|6|99.37% / 99.36%|99.58% / 99.72%|
|7|69.32% / 69.29%|87.24% / 87.21%|
|8|88.40% / 89.34%|96.00% / 96.38%|

Gains are not universal at higher ranks. Head 4 regresses at ranks 16 and 64; head 5 regresses slightly at rank 64, and head 0 has a small positive-sign rank-64 regression. Although the training local quadratic objective is no worse than PCA at every rank, the finite held-out behavior need not follow that ordering. Local nonlinearity and metric sampling error remain competing explanations.

All truncated-map operator norms are modest in this panel (about 1.29–2.12), but oblique amplification is real and must be assessed under distribution shift. Rank 128 reconstructs the full source and native response. The native MLP/baseline remain in the evaluation; this is a source-coordinate experiment, not a stand-alone compressed model.

Audits: double-precision central differences verify the composed logit derivative for one direction per head at two perturbation scales, with maximum relative error about 3.04e-8. Every local weighted objective is no worse than its PCA counterpart; all full-rank source and finite-response endpoints pass. Native effect denominators replay H497. Archived per-case arrays are finite; source basis, covariance, Fisher matrix, and weighted eigensystem are saved. No new documents were used; tracked total is 18,880.

Next H499: freeze the rank-4 construction for every head and evaluate both PCA and Fisher maps on 512 new complete-document-disjoint receiving contexts. Retain the same 64 donor sources to isolate receiver transfer. Report all 18 signed head-specific gain estimates with one simultaneous 95% bootstrap band, conditioning on the frozen maps and donor pool. Resample receivers within each donor assignment, paired across all methods/heads/signs. No head or sign is selected after fresh outcomes. This tests receiver generalization, not fresh-donor robustness or semantic meaning. Fresh donor transfer, local-versus-finite metrics, and matched CE comparisons remain open.
