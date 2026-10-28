## H1044 — Direct change encoding improves edits over differences of sparse endpoints

Previous goal turn was progress: H1043 verified full expanded exposure without consistent recovery and stopped duration extensions. This turn tested whether sparse support should be chosen for the change rather than separately for its two endpoint activations. No fitting or output-dependent support selection occurred.

Geometry and group memberships are frozen from the expanded H1042/H1043run. Direct encoding computes delta=donor-recipient for each of three streams, projects delta/scale into PCA coordinates, chooses support from squared delta coordinates (optionally multiplied by frozen learned importance), decodes once without a center, and adds the result to the native recipient. Endpoint controls decode each endpoint with the old selector and subtract. Both receive the same pair and full native recipient context. This is a paired-edit representation, not autonomous state reconstruction.

Budgets differ in a useful, explicit way: direct selection uses at most16coordinates perstream/48total. Each endpoint uses at most48, but its difference can occupy the union of two supports; observed union maximum88, theoretical96. Thus direct methods improve despite a smaller effective change-support budget, not through extra active coordinates. Dense projection uses128perstream/384total as a capacity reference and is not matched-budget.

Paired relative KL, lower better:

| Method | Same-family | Historical copying/spatial | Weekday development |
|---|---:|---:|---:|
| Endpoint scalar, trained |0.7000|0.9026|0.8911|
| Direct scalar, transferred trained weights |0.4695|0.8635|0.7672|
| Direct scalar, untrained |0.4475|0.8448|0.7498|
| Endpoint block, trained |0.7991|0.9405|0.9087|
| Direct block, transferred trained weights |0.7191|0.9115|0.8910|
| Direct block, untrained |0.6322|0.9166|0.8416|
| Endpoint shared, trained |0.9755|0.9647|0.9417|
| Direct shared, transferred trained weights |0.7933|0.9373|0.8700|
| Direct shared, untrained |0.7161|0.9220|0.8974|
| Dense direct projection |0.2527|0.7000|0.5819|

All nine architecture/panel comparisons of transferred trained direct versus trained endpoint improve. Untrained scalar direct is stronger still; the learned importance weights were optimized for endpoint reconstruction plus endpoint-difference edits, so their transfer to a new input statistic should not be assumed beneficial. These results support choosing coordinates for changes themselves and avoiding wasted support on background activations/support-switching artifacts. They do not isolate background allocation from switching as separate causes; both change together.

Fidelity remains inadequate. Donor-correct top1on copying/spatial is0for every arm, including dense. Same-family donor top1is22.5%untrained direct scalar versus0%endpoint scalar; dense reaches47.5%. Weekday donor top1is12.5%for all arms. These are correctness after paired interventions, distinct from earlier whole-reconstruction top1. Dense projection is a fixed Euclidean reference, not a behaviorally optimal upper bound over every possible encoding in the span.

Interpretation is conditional on the same small opened evaluation panels, single seed, PCA geometry, and fixed random groups. No general block/shared advantage, new-task discovery, or manifold validation established. Nevertheless, the measurement supplies a concrete change in representation objective: local edits can be represented more economically by their own support, while native context remains available. Context-conditioned feature work should not be judged solely by reconstruction of complete activations.

Verification:780suffix evaluations and zero fitted rows. Endpoint controls replay H1043paired KL exactly. All checkpoint/source hashes unchanged. Direct encoding returns exactly zero for zero changes and is exactly antisymmetric on checked reverse-change examples for each arm. Sparse maximum48, endpoint union maximum88. Independently recomputed KL for30saved endpoints (one per panel/arm), maximum3.78e-14. No native output used to choose support.

Next bounded experiment: train direct-change scalar/block/shared importance weights from zero on the same expanded80training pairs, fixed160steps and same family schedule, using only finite paired-edit KL plus the same small importance regularizer. Do not include whole-state reconstruction loss for a representation intended solely for changes. Compare all final checkpoints against frozen untrained direct controls and transferred endpoint-trained weights on the same panels, with no evaluation-dependent stopping. This tests whether supervision helps once the encoded quantity matches the causal objective. If it fails, stop diagonal-importance tuning and investigate learned dictionary/group geometry instead of extending this optimizer run indefinitely.
