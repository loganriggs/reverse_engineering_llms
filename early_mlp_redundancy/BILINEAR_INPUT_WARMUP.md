# Bilinear input warmup: audited optimization failure

M81 local derivative preflight is complete with documented hard-gate crossings. M82 completed all eight fixed warmups and their audits. Seven of eight models became worse on full-training reconstruction than initialization. Downstream causal calibration and the16matched supervision fits have not started. Preserve all states and all data; do not select the successful seed.

## Derivatives before optimization

The first native finite-difference experiment checked32directions at96steps. All endpoint directions passed. Four parameter directions lacked two support-stable steps because the larger perturbations crossed hard selection boundaries; all four agreed at the smallest step. The original result remains marked unresolved. A separately frozen follow-up tested all16parameter directions at two smaller steps with the same tolerance and examples. Every follow-up step was support-stable and agreed with autograd; all gradients replayed the original values. This establishes local derivative agreement, not differentiability across a selection boundary. Natural KL arithmetic also passed independent scoring.

There were1,552native forwards in these two gradient checks, or1,712including the previous architecture/replacement/loss preflight. No optimizer update occurred until the combined completion receipt and prospective warmup/evaluation protocols were frozen.

## Fixed warmup and audit

All8models received exactly1,000Adam updates atlr.001, batch128, using all4,092eligible natural training positions with identical perseed sample streams across architectures. Decoder columns were normalized after updates. No task data or heldout outcomes entered fitting. Independent audit verifies final states, unchanged normalization/membership, allAdamstep counts, perstep sample-stream hashes, and full-training NumPy predictions, routing and MSE. It does not independently replay every optimizer update.

Errors below are MSE in fixed output-normalized units. They use the entire training population, not a selected minibatch.

| Kind | Seed | Initial MSE | Final MSE | Largest position's share of final error |
|---|---:|---:|---:|---:|
| dense | 71 | 1.127 | 0.769 | 0.36% |
| scalar | 71 | 1.113 | 21.970 | 46.19% |
| block | 71 | 1.085 | 30.897 | 91.27% |
| shared_fixed | 71 | 1.064 | 1021.762 | 91.28% |
| dense | 72 | 1.138 | 1.580 | 38.77% |
| scalar | 72 | 1.122 | 241.945 | 97.99% |
| block | 72 | 1.091 | 476.931 | 99.57% |
| shared_fixed | 72 | 1.067 | 94.294 | 98.91% |

The dominant position in seven models is cache index1280, document10/position0, token447. Its normalized input norm is155.62. For shared_fixed seed71, its error rose from83.05 to3,816,416, contributing91.28%of final total error. Other positions also matter: scalar seed71 has a median per-position error2.119, so its failure is not confined to one point. Dense seed71 improves overall despite having large transient training spikes.

These are posthoc training-only diagnostics. A large input and bilinear scaling make amplification plausible, but this does not yet separate sampling, Adam dynamics, encoder gain, decoder renormalization, or hard support changes. All data remain included. A low final minibatch loss can coexist with catastrophic full-population error, so it must not serve as the readiness criterion for downstream training.

## Course correction and next bounded diagnostic

Do not calibrate causal weights or launch downstream fits from this failed warmup as if it succeeded. First replay two existing trajectories exactly: shared_fixed71 (largest final failure) and dense71 (only full-population improvement). The posthoc diagnostic selection is explicit and is not model selection for evaluation. Record the rare position's error, branch gains, support, gradient and update norms around its scheduled appearances, while verifying the original sample stream and final state. This changes instrumentation only, not optimizer, data or architecture. Stop after one replay per selected trajectory and use it to specify a controlled stabilization experiment, retaining the original runs as evidence. No learning-rate or architecture sweep is authorized by this diagnostic.

The overall aim remains executable, reusable input features and a matched causal-supervision comparison. Warmup failure limits what this particular optimization setup establishes; it does not establish that blocks, shared atoms or nonlinear features are unsuitable for the model.

Artifacts: validation/bilinear_input_preflight_completion.json, bilinear_input_warmup_protocol.json, bilinear_input_prospective_evaluation.json, bilinear_input_warmup_results.json, bilinear_input_warmup_audit.json, bilinear_input_warmup_error_diagnostic.json, and the original/follow-up native gradient records. Large model/optimizer states remain in the research artifact directory.
