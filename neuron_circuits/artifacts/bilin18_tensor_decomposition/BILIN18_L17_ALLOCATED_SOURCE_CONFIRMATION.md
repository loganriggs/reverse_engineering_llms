## H510 — Fresh-document confirmation of training-based rank allocation

Freeze H509's total36-coordinate allocation [10,3,16,1,1,1,1,1,2] across heads0–8. Compare with uniform4-per-head full-path Fisher maps and uniform4-per-head PCA. Evaluate joint replacement of each document's own attention-source contributions on512 new complete-document-disjoint contexts. Both MLP input and direct residual path use the reconstructed attention write. All source means, covariances, metrics, maps, and ranks remain frozen; there is no outcome-based selection or refit.

Two primary comparisons were prescribed before outcomes: reduction in mean KL versus uniform full-path Fisher and versus PCA. Use4,000 paired iid document bootstrap samples and one simultaneous95% band over both gains. Cross-entropy is descriptive, not a co-primary improvement test. Mean replacement retains frozen calibration means and removes all centered attention-source variation.

| Reconstruction | Mean KL to original output | Recovery vs mean replacement | Observed-token ΔCE (paired SE) |
|---|---:|---:|---|
|mean|0.012111|0.00%|+0.012326 (0.008329)|
|PCA|0.005950|50.87%|+0.011190 (0.006040)|
|full_metric|0.006048|50.06%|+0.012150 (0.006163)|
|allocated|0.004431|63.42%|+0.009818 (0.004927)|

| Allocated map versus | Mean KL reduction | Simultaneous95% interval | Positive lower bound |
|---|---:|---|---|
|full_metric|0.001617|[0.001118, 0.002117]|True|
|PCA|0.001519|[0.001145, 0.001893]|True|

Both prespecified comparisons pass. The allocated map reduces KL by26.74% relative to uniform full-path Fisher and25.53% relative to PCA. Its mean KL is0.004431, versus0.006048 and0.005950 respectively. Recovery relative to mean-source replacement is63.42%, compared with50.06% and50.87%. This is not63% accuracy or preservation of63% of the model's entire computation; it is a ratio to a specified attention-write replacement perturbation.

The confirmation supports choosing source capacity according to a training-data/downstream-metric criterion rather than imposing an equal rank on each architectural head. The fixed allocation transfers to new natural contexts from a later corpus segment. It does not prove the allocation is unique, stable under all distribution shifts, semantically interpretable, or globally optimal for joint finite KL. Uniform full-path Fisher is not better than PCA on this panel; the validated benefit is the allocated construction as a whole.

Observed-token CE increases are positive in this panel (allocated +0.009818 nats with paired SE0.004927). They are not equivalent to KL, and no CE equivalence or improvement was tested. Do not describe the compression as lossless or task-loss neutral. The original attention/head activations and full native MLP are still computed for evaluation; no deployment parameter/FLOP reduction is demonstrated.

Data handling: the old canonical cache had only498 unused eligible complete documents, so its512-document builder stopped before saving a panel. A new10,000×513 uint16 FineWeb cache was built from stream documents73000–80423 using the same GPT-2 tokenizer snapshot. The builder saved the array and manifest, then aborted during Python interpreter finalization (exit134). The saved array was independently validated for shape, dtype, token range, and manifest SHA256 aa3e7960719f73b659e5a987e1d21dea72907e3b690358763e9dfdf7601cf70b; it was not regenerated or accepted solely on process intent. The new panel uses seed20271226.

The independent document audit rehashed all19,456 excluded documents and512 new documents against each manifest's own source cache, verified token windows, and found zero overlap. Total tracked documents is19,968. This is disjointness for the tracked research continuation, not a model-pretraining holdout claim. Initial runner session70936 terminated with an indentation-related NameError before main or evaluation; only indentation was corrected before session2243 ran successfully. No scientific setting or outcome-dependent criterion changed.

Audits: frozen artifacts were hash-checked before execution; head writes sum to the attention write with relative error4.38e-7; next-token alignment is asserted; arrays are finite and paired mean-KL gains replay. The shared document bootstrap critical value is2.1368. Confidence intervals condition on the frozen fits/calibration and use the sampled document population; they do not include representation-training uncertainty. All fresh source writes, receiver states, labels, and per-case results are archived.

Next H511 asks whether architectural head boundaries are useful constraints for a data-based canonical form. Construct a single rank36 representation of the summed centered attention write, using the same original source-calibration documents and the same64 training metric receivers. Compare joint-source PCA36 and full-path-metric rank36 maps against the frozen per-head allocated36 construction. Preserve the summed calibration mean. Compute a joint covariance and full-path metric in its supported calibration subspace, retain covariance rank/spectrum and held-out outside-span energy as explicit limitations, and evaluate on the now-reused H510 panel as a diagnostic. Joint covariance rank is limited by calibration sample size; do not treat its empirical nullspace as a universal functional nullspace. This can separate the benefit of cross-head correlation modeling from the benefit of probability weighting, without selecting ranks after evaluation. If promising, confirm a frozen construction on another fresh panel before making generalization claims.
