## H705 — Early-layer projection floors constrain the sparse-rotation pilot

Return to the existing H681 model activations: value0and post-block residual0/1/2, each1152dimensional. Fit mean and centered covariance on2048training documents (32768vectors), using float64 eigendecomposition. Evaluate projection on256validation documents (4096vectors). The final256document test split remains untouched. No new model forwards or documents.

| Site | Train rank for90%variance | Validation loss at128D | Validation loss at256D | Validation loss at512D | Prior signed rank1BSF loss,32active scalars |
|---|---:|---:|---:|---:|---:|
|value0|336|36.295%|16.874%|3.416%|23.144%|
|residual0|435|27.852%|18.002%|8.187%|22.174%|
|residual1|494|36.143%|22.854%|9.886%|27.887%|
|residual2|567|40.811%|26.898%|12.609%|31.249%|

A128dimensional pilot would already lose27.9–40.8%of residual validation variance before any sparsification. At256dimensions, residual projection losses are18.0–26.9%; value0loss is16.9%. Full90%training variance needs336dimensions for value0,435for residual0,494for residual1and567for residual2. Thus very small rotations cannot stand in for a faithful full model representation.

A fixed PCA span needs approximately192–208dimensions to match the historical rank1signedBSF validation error, despite that BSF using only32active scalar coordinates per example from a2304atom dictionary. This is not a fair-budget superiority test: the BSF chooses different directions per input, whereas PCA uses one fixed span. It highlights why retained rank and per-example active dimension must remain distinct. Historical BSF statistics use float32 centering/normalization and are contextual references; no BSF reevaluation or new checkpoint selection occurred here.

Whitening at256dimensions has a maximum-to-minimum scale ratio of4.5for value0,18.5for residual0,10.0for residual1and8.3for residual2. The full value0span is far more ill-conditioned (ratio about22316), so blindly whitening all directions could amplify nearly-null variation. These diagnostics motivate a bounded retained-span pilot rather than assuming the synthetic full-rank procedure transfers unchanged.

Next fit256dimensional whitened sparse rotations on value0(the token-only control) and residual1(the contextual site with prior causal evidence), three seeds. Compare PCA-coordinate, random-rotation and learned-rotation controls within the same span. Evaluate sparse reconstruction at several active-coordinate budgets, separating irreducible projection loss from within-span sparsification error. Assess cross-seed stability and token-versus-context coactivation before proposing manifold labels or model interventions. No synthetic four-group/three-atom ontology will be imposed on the model data. This remains an exploratory pilot, not a BSF replacement or full account of model computation.

Audits: PCA eigenvectors orthonormal,128Dtraining whitening checked, direct128Dprojection residual agrees with energy-accounting formula within1e-10. All source hashes unchanged, finite results, test evaluations0, whole-model forwards0. Saved training eigensystems support the next run without refitting on validation. Broad goal active.
