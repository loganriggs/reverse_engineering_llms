## H491 — Fixed-rank probability-oriented output subspace

Fix source rank4 and cross-output rank64. Initialize at H490's data-weighted cross-term SVD basis. Optimize only the orthogonal output projector for actual finite full-source KL, keeping source axes, scalar normalization, source quadratic core and native baseline write fixed. Coefficient reads for any proposed basis remain exactly M_i^T V^T; there is no separate learned coefficient map. The full-source target includes the omitted source tail, so the objective can encourage compensation for correlated tail effects as well as better cross-term retention.

Use the same1024 training receivers/32 donors, both signs. Take16 full-batch steps of the horizontal Grassmann gradient g−(g V^T)V. Normalize a proposed update by the first-order change in projected training cross responses. QR retract candidates at fractions [.001,.01,.03,.1] to orthonormal rows; include the unchanged model. Choose the candidate with lowest training KL and choose stopping point on256 validation receivers/64 donors, including step0. This is a finite optimization trajectory without a convergence guarantee or claim of global KL-optimal subspace.

Validation selects step0: every attempted update has worse validation KL than the initialization, despite decreasing training loss. Therefore the selected model is exactly the original data-SVD model. Freeze the selection before collecting outcomes on512 new document-disjoint receivers. Independent audit rehashes18368 excluded documents and512 new ones, verifies windows and finds zero complete-document overlap. Total tracked documents18880. Donor family/calibration unchanged; not a model-training holdout.

Initial training/validation KL: 0.0000843949/0.0001131536. First step: 0.0000795482/0.0001142171. Final step16: 0.0000652620/0.0001194383. Selected step: 0.

| Predeclared primary statistic | Estimate | Joint95% interval | Threshold | Pass |
|---|---:|---|---:|---|
|absolute_negative|0.9604|[0.9417, 0.9790]|0.95|False|
|metric_gain_negative|0.0000|[0.0000, 0.0000]|0|False|
|absolute_positive|0.9612|[0.9387, 0.9837]|0.95|False|
|metric_gain_positive|0.0000|[0.0000, 0.0000]|0|False|

All four gates fail. Selected and initial predictions are identical, so their paired gains and intervals are exactly zero—not evidence that all probability-oriented bases are equivalent. The unchanged model reaches96.04/96.12% full-source KL recovery on this panel, versus94.14/94.51% on H490's different panel. That difference is receiver sampling, not improvement from optimization. Joint absolute lower bounds94.17/93.87% still fail>.95. Raw response recovery is99.31/99.28%.

The bootstrap uses4000 paired within-donor receiver resamples and joint4 standardized bands, conditioning on fitted model, calibration, donor pool and selection. Zero-variance paired-gain coordinates are retained as exact zero intervals and excluded from standardization; otherwise division by zero would corrupt the bands. Training-sample uncertainty is not included. Validation reuse and the large number of possible output-basis orientations limit conclusions about optimization generalization.

Audits: initial validation recoveries exactly replay H490; QR candidates orthonormal within1e−10; training loss nonincreasing; arrays finite; selection/compiled maps saved before confirmation; independent document/window audit passed. Initial and selected compiled read/write maps, source basis, quadratic core, complete line-search history and per-case outcomes saved. Conditional scalar count stays391680 array entries for one source4/cross64 model, excluding native baseline computation and scalar epsilon. No process remains active.

Interpretation: this probability-oriented rotation trajectory offers no validated alternative to the data-weighted SVD basis. It does not establish that energy is the correct universal metric or that a better regularized/sampled optimization could not help. Together with earlier failed generic refinement, it argues against repeatedly treating minor objective optimization changes as the only bottleneck. The structured computation itself remains supported, but stable predictions are different from stable feature coordinates.

Next test canonical-form stability. Recompute source4/cross64 output SVDs from disjoint, donor-balanced halves of the existing training receivers, using the same fixed source calibration and exact operators. Compare subspaces using principal angles and projection overlap, and compare individual axes only with sign/permutation alignment while inspecting singular-value gaps. Evaluate both conditional programs on the same already collected receiving panel, comparing prediction discrepancies and native fidelity. Similar behavior with unstable axes would support a stable functional equivalence class without unique feature directions; stable subspaces/axes would strengthen a data-based canonical description. This is a diagnostic of sampling sensitivity, not a new confirmation or proof of global uniqueness. Broader semantics, observed-token CE, native-baseline removal and cross-head/layer transfer remain open.
