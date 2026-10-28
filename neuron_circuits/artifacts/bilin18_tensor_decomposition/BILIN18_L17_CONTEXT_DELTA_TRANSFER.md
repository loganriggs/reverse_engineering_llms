## H531 — Context-dependent updates are harder to preserve than whole source writes

For each document and short history L=32/64/96, define delta=s_128−s_L using summed L17 attention writes. Hold the native128-token pre-MLP receiver fixed and substitute the source write s_L+A(delta). The resulting pre-MLP input is p_128−(delta−A(delta)), evaluated through the full residual bypass plus native MLP and readout. A is a frozen H522 PCA/Fisher map at rank36/144/576. The old reconstruction mean cancels and is not inserted.

Zero-map removal replaces the long-context source with the short-context source while keeping the long-context receiver's other components. Identity preserves the complete update and must reproduce native predictions. This is an artificial hybrid intervention on the final attention write, not the actual short-context model or total contribution of extra context. Earlier residual computation and receiver sensitivity remain long-context quantities.

| Short history | Zero-update KL | PCA36 recovery | Fisher36 recovery | PCA144 recovery | Fisher144 recovery | PCA576 recovery | Fisher576 recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
|32|0.00276475|57.23%|60.76%|82.25%|84.01%|97.65%|98.20%|
|64|0.00117808|58.52%|60.93%|82.65%|84.07%|97.58%|98.14%|
|96|0.00041897|58.32%|60.31%|81.68%|83.18%|97.45%|97.99%|

Fisher36 recovers about60.3–60.9% of the contextual update effect, versus72.4% for whole-source compression at the same128-token receiver in H530. Fisher144 recovers83.2–84.1%, versus88.8% for the whole source; Fisher576 recovers98.0–98.2%, versus98.9%. These ratios have different removal baselines and are not an additive decomposition of performance, but show that the frozen low-rank maps are less effective for this update distribution.

The raw geometry agrees with that limitation: Fisher36 retains about56.7–59.1% of delta energy, while PCA36 retains60.1–62.3%. Fisher nevertheless has better behavioral recovery for all tested ranks/gaps. Behavioral weighting helps, but does not make the contextual update a tiny-rank object in this fixed representation.

The zero-update KL shrinks from0.002765 for32→128 to0.001178 for64→128 and0.000419 for96→128. Consequently, similar recovery percentages refer to different absolute effects; smaller gaps are not evidence of equal contextual importance. All three zero-map effects are nonzero. Identity KL and CE changes are exactly zero.

| Short history | Map | Prediction KL | Raw delta recovery | CE change ± SE |
|---|---|---:|---:|---:|
|32|zero|0.002764752|0.00%|+0.003177 ± 0.001966|
|32|PCA36|0.001182605|61.28%|+0.001410 ± 0.001316|
|32|Fisher36|0.001084991|58.19%|+0.000748 ± 0.001296|
|32|PCA144|0.000490745|82.27%|+0.000245 ± 0.000743|
|32|Fisher144|0.000442169|79.92%|+0.000325 ± 0.000752|
|32|PCA576|0.000065079|97.64%|-0.000318 ± 0.000264|
|32|Fisher576|0.000049711|97.12%|-0.000479 ± 0.000233|
|64|zero|0.001178077|0.00%|+0.001288 ± 0.001153|
|64|PCA36|0.000488709|62.29%|+0.001081 ± 0.000757|
|64|Fisher36|0.000460227|59.09%|+0.000938 ± 0.000754|
|64|PCA144|0.000204426|82.87%|+0.000314 ± 0.000466|
|64|Fisher144|0.000187694|80.51%|+0.000270 ± 0.000472|
|64|PCA576|0.000028495|97.71%|-0.000154 ± 0.000169|
|64|Fisher576|0.000021925|97.20%|-0.000133 ± 0.000148|
|96|zero|0.000418966|0.00%|-0.001026 ± 0.000662|
|96|PCA36|0.000174636|60.12%|-0.000128 ± 0.000443|
|96|Fisher36|0.000166273|56.72%|-0.000166 ± 0.000441|
|96|PCA144|0.000076769|82.13%|-0.000366 ± 0.000281|
|96|Fisher144|0.000070469|79.65%|-0.000461 ± 0.000278|
|96|PCA576|0.000010676|97.62%|-0.000105 ± 0.000100|
|96|Fisher576|0.000008409|97.09%|-0.000074 ± 0.000092|

This supports the hypothesis that whole-source compression preferentially preserves shared activity and gives an optimistic picture of contextual-update retention. It does not prove the retained directions are context-independent or identify semantic features. Differences could reflect source covariance, nonzero delta means, receiver sensitivity, or their associations. Descriptive CE changes are not task-improvement claims. All2,048 documents are reused; no fresh inference, rank choice or subgroup selection is introduced.

Audits: frozen model/source hashes checked; same target across histories verified; no reconstruction mean added to deltas; identity controls reproduce native KL/CE exactly; all per-document measurements finite. Error and delta energies and per-document KL/CE are archived. No new documents:26,112.

Next H532: fit update-specific maps using only the existing3,072 source-calibration documents (the H5141,024 plus H5222,048), disjoint from H527 evaluation documents. Collect matched final-token source writes for32- and128-token histories ending at input token127, and form training delta32→128. Keep the old1,023-dimensional supported basis, original1,024-context Fisher metric and ranks36/144/576 fixed. Fit the uncentered delta second moment C_delta=E[delta delta^T] within that basis, because reconstruction has no intercept and the delta mean must be preserved. Build covariance-PCA and Fisher maps with full matrix square roots/inverses, verify positivity/idempotence, and save before evaluation. Compare these maps to frozen whole-source maps on all three H531 update gaps, reporting all ranks and both methods. No evaluation delta is used to fit. The fixed metric was trained at position63, so residual mismatch remains possible and must be disclosed. This tests whether the information bottleneck is the chosen calibration objective/distribution rather than a universal set of source coordinates. Results are reused-panel diagnostics and need fresh confirmation for superiority claims.
