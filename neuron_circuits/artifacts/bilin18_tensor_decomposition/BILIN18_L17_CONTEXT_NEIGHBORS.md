

## H311 — Joint input geometry predicts missing-response direction

Reference first768 discovery documents; query final256 documents. Same18 head/payload groups and four exact positions. Reference neighbors are restricted to the same group and position, guaranteeing different reference/query documents. Context descriptor is RMS-normalized input x minus its reference-only mean, then unit normalized. Intervention descriptor is unit-normalized z−x, where z is the normalized edited state. Scores are context cosine, intervention cosine, or their equal average. Fixedk1/16. Predict by averaging reference unit missing-response vectors and renormalizing; evaluate signed cosine to query unit response. Random reference controls use fixed same-position indices. Neighbor selection never uses query responses. No semantic labels or fitted distance weights.

| Selector | Actual k1 | Actual k16 | Angle k1 | Angle k16 |
|---|---:|---:|---:|---:|
|random|0.0024|0.0087|0.0005|0.0028|
|context|0.0992|0.1359|0.0228|0.0452|
|delta|0.0885|0.1327|0.0135|0.0265|
|joint|0.1536|0.2099|0.0719|0.1249|

| Contrast k16 | Family | Mean cosine gain | Paired document95% |
|---|---|---:|---:|
|joint_direction_gain|actual|0.2012|[0.1968,0.2054]|
|joint_over_context|actual|0.0740|[0.0720,0.0761]|
|joint_direction_gain|angle|0.1221|[0.1196,0.1245]|
|joint_over_context|angle|0.0797|[0.0780,0.0813]|

Both locked gates PASS: joint k16 exceeds random by>=.05 cosine in both families with positive paired95% lower bounds; it exceeds context-only by>=.02 in both families with positive lower bounds. Bootstrap2000 resamples of256 query documents preserves heads and four positions; intervals condition on fixed references, fitted residual approximation and payload draws. These are reused-validation results, not fresh confirmation.

**Inference.** Missing responses have predictable directional variation conditioned on context AND intervention geometry. This is compatible with a bilinear relational computation despite poor global output-PCA compression: a broad union of directions need not be unpredictable. Joint matching outperforms either single descriptor, supporting use of both inputs to local prediction. It does not establish semantic features, a small computation graph, or unexpected structure beyond generic bilinear smoothness. Mean cosines.210/.125 remain far from accurate direction reconstruction, and cosine is not raw-response fidelity or downstream KL.

Retrieval stores many native-supervised reference outputs and is not an economical compiled model. Neighbors are selected within a head/payload class; cross-head transfer was not tested. Reference/query document hashes are disjoint within the original panel, but the residual approximation already used reference data and these query documents have been examined repeatedly. Model-training or external-distribution holdout is not claimed.

**Audit.** Saved indices are all within the3072 reference positions, match query position modulo4, and have1024×16 shape. Summary means replay from saved per-document rows. Cached-response source hash checked. An initial8-row replay failed its numerical tolerance; after matching the original256-row computation chunk, replay passed below1e-5. The failed log is preserved. This is consistent with float32 batch-shape sensitivity; no threshold was loosened. The replay verifies one256-row slice, not every cached response. All jobs terminal; native weights unchanged.

**Next discriminator.** Freeze this retrieval rule and test new documents, adding a simple combination of independently context-selected and intervention-selected predictions as a control. This distinguishes benefit from joint neighbor selection from merely combining two predictors. Report signed direction cosine AND raw-response reconstruction with amplitudes predicted solely from references; no oracle target norms. Fresh confirmation should precede semantic labeling or a larger chart-fitting sweep. Broad research goal active.
