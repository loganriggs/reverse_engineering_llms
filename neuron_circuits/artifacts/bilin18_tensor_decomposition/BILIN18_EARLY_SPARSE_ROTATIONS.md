## H706 — Sparse rotations improve same-span reconstruction on real activations

Fit256dimensional whitened sparse rotations on existing value0and residual1training activations, using H705's training-only mean/eigensystem. Three seeds per site,2000Adam steps, batch512, learning rate0.003. Select by validation mean sqrt(whitened coordinate²+1e-6), every200steps. Compare PCA coordinates and initial random rotations within the identical retained span. No group ontology or semantic labels are fitted.

At evaluation, rank coordinates by individual decoded-contribution norm, retaining8/16/32/64/128/256. Decoder directions after unwhitening are generally nonorthogonal, so this TopK rule is a stated heuristic, not optimal subset reconstruction. Report total centered activation error as projection floor plus within-span sparsification error. Per-example coordinates and dictionary width differ from earlier BSFs; these are not matched-resource claims.

| Site | Coordinates | Mean error,32active | Mean error,64active | Mean error,128active | Projection floor |
|---|---|---:|---:|---:|---:|
|value0|pca|47.02%|32.85%|20.85%|16.87%|
|value0|random|58.57%|40.05%|22.90%|16.87%|
|value0|learned|38.32%|28.36%|19.69%|16.87%|
|residual1|pca|44.76%|34.36%|25.72%|22.85%|
|residual1|random|60.56%|43.77%|28.25%|22.85%|
|residual1|learned|40.17%|32.58%|25.39%|22.85%|

At32active coordinates, learned rotations reduce error from47.02%to38.32%for value0and44.76%to40.17%for residual1relative to PCA-coordinate truncation. They beat the random-rotation controls by larger margins. Validation smooth absolute coordinate magnitude also decreases: value0from0.696PCA/0.742random to0.519learned; residual1from0.763/0.769to0.642. These are evidence of useful sparse coordinate organization in the retained data span, not semantic manifolds or a recovered computation graph.

The historical overcomplete signed rank1BSF at32active scalars has lower validation error:23.14%value0and27.89%residual1. This pilot does not replace that baseline. Its256dimensional fixed span already discards16.87%and22.85%respectively; it cannot express arbitrary1152dimensional directions outside that span. The pilot's larger gain on the token-only control also cautions against interpreting sparse rotations as contextual computation by default.

Next assess cross-seed repeatability of decoded directions and active supports against random-rotation controls, then quantify token identity versus context in stable components. Candidate groups should be driven by recurring coactivation/subspace evidence, not copied from the synthetic four-group setup. Only then proceed to matched-norm interventions and diverse task-specific model-output evaluations. The user-requested manifold/computation investigation is now back on actual model activations, while the stronger BSF baseline remains intact.

Runtime: an initial GPU attempt completed one fit then terminated with a CUDA launch error. That attempt was discarded; the runner was updated to save completed fits, and a synchronous-CUDA retry completed all six. No failure was treated as a successful fit. Final source hashes unchanged; finite metrics; maximum rotation orthogonality error1.33e-5. Independent float64 direct reconstruction checks on one learned seed per site at32and256active coordinates agree with reported decomposition within6.23e-8. Test evaluations0, whole-model forwards0, no new documents. Validation remains exploratory and the reserved test split stays untouched. Broad goal active.
