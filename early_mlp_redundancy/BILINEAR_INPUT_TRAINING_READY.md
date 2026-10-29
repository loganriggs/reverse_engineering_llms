# Full-population warmup succeeds; causal supervision is ready

M85 and M86 are complete. The optimizer diagnostic series is closed. All eight full-population warmups pass independent audits and improve training reconstruction; the single global causal-loss weight is calibrated and audited using training data only. The16downstream supervision fits have not started, and no held-out quality claim follows from these results.

## Controlled full-population comparison

The intervention replaces random128-position gradients with the uniform gradient over all4,092training positions on every update. All original initial states, Adam defaults,lr=.001,1,000updates, decoder renormalization, data and objective remain fixed. This uses31.96875times as many examples per update; it is not an equal-compute comparison. It changes both gradient variance/intermittency and data exposure. It does not isolate those two effects separately.

All eight final training MSEs are below initialization and the mean-write reference1. Independent NumPy predictions/gates match recorded full-training losses; buffers, fullpopulation index hashes, finalstates and allAdamstep counts are checked. No individual position dominates final error: the largest position contributes only about0.10–0.13%of total error. The losses are not strictly monotonic, but the catastrophic earlier spikes disappear: the highest recorded full-objective value across all runs is about1.307.

| Kind | Seed | Full-training normalized MSE | Mean active coordinates |
|---|---:|---:|---:|
| dense | 71 | 0.3617 | 128.00 |
| scalar | 71 | 0.4132 | 32.00 |
| block | 71 | 0.4273 | 32.00 |
| shared_fixed | 71 | 0.4664 | 27.71 |
| dense | 72 | 0.3625 | 128.00 |
| scalar | 72 | 0.4140 | 32.00 |
| block | 72 | 0.4260 | 32.00 |
| shared_fixed | 72 | 0.4575 | 27.94 |

Dense uses128coordinates and sparse models32or fewer, so this ranking is not a matched-active-cost result. A reconstruction difference is not a difference in causal fidelity or interpretability. The evidence supports using these eight states as learning starting points under an explicitly recorded amendment, not declaring them recovered computational units.

## Calibration and matched comparison

All eight starts are retained; the failed stochastic and no-momentum runs remain archived. The starting-state amendment was frozen before calibration and uses only training readiness, not held-out performance.

Calibration uses all1,056scheduled training model/assignment/endpoint rows, including repeated scheduled examples and weak cases. Mean intact task KL is0.67771036; mean balanced three-corner response error is2.61667296. Their ratio gives one global weight0.25899697. There is no architecture-specific scaling or evaluation-driven tuning.

The independent audit rescored9,480response vectors and1,056intact predictions, including3,168exact-zero identity vectors. It verifies each response's own background, teacher references, family/region scales, schedule/model inventory and global arithmetic. Maximum scalar discrepancy is1.07e-13. Calibration required7,368native forwards and no optimizer update.

Next are16fixed fits:4architectures ×2seeds ×2objectives (intactKL versus intactKL plus the weighted three-corner response). Within each architecture/seed, both arms share the exact warmup state and132-update schedule, with66natural and66task updates, both endpoint losses averaged, and Adamlr1e-4. No optimizer state carries over. No additional warmup search is planned. Retain all final states, then evaluate every model on the predeclared natural, source-group and whole-task holdouts and privileged output-projection references.

The central experiment is now again the requested one: whether causal supervision helps executable input features capture reusable computation. Manifold validity, transfer to excluded tasks and genuine new-task discovery remain distinct claims requiring evidence.

Artifacts: validation/bilinear_input_fullbatch_results.json, bilinear_input_fullbatch_audit.json, bilinear_input_fullbatch_error_diagnostic.json, bilinear_input_joint_calibration_protocol.json, bilinear_input_joint_calibration.json, bilinear_input_joint_calibration_audit.json, and bilinear_input_joint_training_protocol.json.
