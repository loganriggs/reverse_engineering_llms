## H1035 — Residual projection drives held-out failure, with cached-value interaction

Previous goal turn was progress: H1034 independently audited frozen selector transfer and redirected work away from further gate tuning. This turn completed the prespecified eight-corner projection experiment. No training, rank search, or candidate selection occurred.

Panel: the same 16 historical pairs/32 endpoints from H1034's four native-competent copying and spatial-inversion cohorts. Both entire task families remain excluded from the PCA training. These outcomes now serve diagnosis, not untouched validation. Each input stream is either native or projected into its frozen discovery-trained 128-dimensional PCA subspace. Streams are the residual immediately before attention 11, the initial embedding residual input, and cached first-layer values. All downstream layers are native.

For whole-state reconstruction, replace selected streams with their projections. For paired interventions, apply recipient plus the hybrid donor-minus-recipient representation; target is full donor output. Thus native corners preserve all three full donor differences. Lower KL is better. Paired relative KL is normalized to the no-edit divergence and is not an understanding percentage.

| Projected streams | Whole-output KL | Paired relative KL | Intended top-answer accuracy |
|---|---:|---:|---:|
| None |0.0000|0.0000|90.6%|
| Cached values |0.0574|0.0487|93.8%|
| Initial embedding |0.0109|0.0001|93.8%|
| Embedding + cached values |0.0623|0.0497|100.0%|
| Residual |2.0246|0.4787|3.1%|
| Residual + cached values |2.7997|0.7382|0.0%|
| Residual + embedding |2.0781|0.4818|6.3%|
| All three |2.8755|0.7392|0.0%|

Metrics average equally over four family/split cohorts. Small apparent accuracy improvements represent only one to three endpoints and are not evidence for a better model. The residual-only intervention is damaging on both tasks: paired relative KL is 0.3402 for copying and 0.6173 for spatial inversion. Cached-value-only projection yields 0.0953 and 0.0021 respectively. The initial-embedding port is nearly irrelevant to this comparison; this does not imply embeddings are unimportant to the model's earlier computation.

Interaction matters. Adding cached-value projection to residual projection increases paired error by 0.2595, versus 0.0487 when the residual is native. The difference of differences is approximately 0.2108 in this normalized KL metric. That demonstrates non-additive sensitivity under these finite interventions; it does not identify a unique underlying bilinear mechanism or imply independence at other contexts.

Mean centered activation energy retained is 35.3% residual, 50.6% embedding, and 46.9% cached values. Mean paired-difference energy retained is 21.1%, 13.3%, and 19.5% respectively. These are averages of per-example/per-pair norm ratios over full sequences. Similar low difference-energy retention produces very different behavioral damage across streams, reinforcing why Euclidean reconstruction alone is not a sufficient target. The poor residual coverage is substantial, not merely a tiny low-energy tail. This result concerns the particular discovery-trained PCA span and cannot establish that all low-dimensional residual representations fail.

Verification: 512 suffix executions, no fitting. Native logits replay exactly; all-native donor patches differ from saved donor logits by at most 6.68e-6 due to float32 subtraction/addition. All-projected metrics reproduce H1034 exactly. Frozen source hashes unchanged. Independently recomputed saved-logit KL for one endpoint in every arm (eight samples), maximum discrepancy 3.94e-14. This is an independent metric audit plus native/cross-experiment execution checks, not a second independent implementation of the entire suffix.

Featurizer deliverable: preserve a native residual bypass as a diagnostic control and require adequate residual task-difference coverage before treating gate selection as the main bottleneck. Evaluate cached values jointly with residual compression because isolated fidelity overestimates combined fidelity. Do not interpret a frozen random grouping of a narrow PCA dictionary as a broad test of block manifolds.

Stopping criterion met for this diagnostic: all eight fixed corners evaluated and the main damage localized. Next return to training-data coverage: compare training-only residual bases fitted from the narrow task panel versus diverse existing natural-language activations, at matched rank and information access, with activation-only and paired-causal criteria reported separately. Audit existing capture provenance first; do not fit on copying/spatial holdouts or reuse their outcomes to select rank. A separate whole-task evaluation panel is required before claiming new generalization. If suitable natural captures are unavailable, collect a fixed training-only panel rather than silently mixing evaluated tasks into training. The broader manifold/discovery goal remains unresolved.
