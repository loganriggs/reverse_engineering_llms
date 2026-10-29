# Toy research continuation — active, 2026-10-29

Goal: complete the toy experiments, scale them as justified, and derive/test interesting theoretical results. The prior native-MLP research remains paused. This project is separate and must not be redirected automatically into the old featurizer loop.

Previous goal turn: progress (regression controls, d=8 study, quotient geometry). Current goal turn: progress (readout audits, 10x optimization extension, singular-capacity theory and experiments). All current producers/auditors are terminal and passed. No blocker or live run.

## Evidence current through this batch

- Latest report: READOUT_AND_OPTIMIZATION_RESULTS.md; theory: SINGULAR_READOUT_GEOMETRY.md.
- results/readout_refits: 1,400 readouts, 700 Gaussian population projections; independent direct-feature/metric/Pythagorean audit passed.
- results/longer_training: 60 trajectories at N256/1024 × 600/1800/6000 steps = 180 snapshots; all 60 old 600-step states replay exactly; all 180 quadrature and floor checks pass.
- results/singular_feature_paths: all 50 width-matched untied initial-to-600-step hidden parameter paths × 7 epsilon values = 350 records; pivoted-QR/linear-solve audit passes; analytic Hessian/noise toy checks pass.
- Historical: initial README/MATHEMATICS/RESULTS; REGRESSION_AND_D8_RESULTS.md; POLYNOMIAL_QUOTIENT_GEOMETRY.md. All old outputs remain immutable. No GPU work/native-MLP research resumed.

## Findings that change the next action

Training improves fixed hidden-span floors in147/150 cases. At N1024, refitting top weights adds little; hidden-span floors account for most remaining error. Extending training to6000 steps improves population MSE in58/60 trajectories and markedly improves representations. Neither the600-step nor6000-step results prove convergence/capacity limits. No general tying advantage emerges; do not keep extending this same single-output optimization sweep to seek a desired winner.

Unregularized span floors need conditioning/norm controls. In all50 untied parameter paths, rank36 at exact equal branches becomes64 byepsilon1e-5, but median exact-fit readout norm~76379 and ridge benefits negligible. Verified toy(u,u+epsilon*v) explains discontinuous ideal span capacity, weight blowup1/epsilon, Hessian small eigenvalue~epsilon^2/2, independent feature-noise amplification2sigma^2/epsilon^2; common-mode noise cancels. These are mathematical/diagnostic results, not an RLCT estimate or proof of the optimizer's specific mechanism.

## Next bounded stage (not started)

Move toward reusable computation across tasks and E3 observability, rather than another single-output fit. Design a fresh multi-output teacher family where several task heads reuse one hidden dictionary, preserving whole-task head holdouts. Compare shared and independent task trunks with width/parameter controls, random and oracle features, and a simple multitask polynomial baseline. Evaluate few-label fitting of unseen heads, Gaussian/OOD error, and conditioning-aware subspace recovery. Output-function subspace recovery is NOT literal graph recovery.

Before claiming actual graph recovery, explicitly define observations/equivalence: output-only data cannot distinguish a reused state from identical recomputation with tied parameters. A small controlled DAG family with interventions on internal states can test that distinction; state which access identifies physical fan-out and which equivalences remain. Keep parameter tying, repeated inputs, and cached/intermediate execution reuse separate; their numerical gradients may coincide but execution cost/intervention responses can differ.

Freeze the next protocol before outputs. The current data are reused development panels. Use new teacher/data seed ranges and separate teacher from optimizer seeds when confirming learning/transfer claims. Preserve original CPU-first→GPU scaling direction; deeper/wider experiments remain required when informative. Do not overwrite or rerun completed outputs.

## Remaining scope — not complete

- E1: independent teacher/initialization factorial confirmation if an interpretable advantage emerges; original-script corrected setting is only partially matched, not a literal replication.
- E2: additional same-function geometry controls, learning-budget versus convergence distinctions, and useful function-subspace recovery metrics. Existing exact projected-SGD equivalence is complete, not an optimization theorem.
- E3: an explicit identifiable graph family or observable equivalence class, controlled parameter-only/input/intermediate reuse, graph complexity penalties, and intervention/held-out-composition tests. Exact function recovery alone is insufficient.
- Theory: small fiber/critical-locus calculations and explicit likelihood-based singularity work can extend the current notes. Do not infer RLCT from Hessians or claim literature novelty without a source review.
- E4: larger input/width/depth CPU→GPU runs when informative. Respect existing GPU workloads; do not scale just to spend compute or treat loss failures as structural results.
- Track sample error, population function error, parameter count, arithmetic execution cost, and graph recovery separately. The entire requested research program is not achieved.

Runtime: /home/loganriggs/Coding/neuron_circuits/.venv/bin/python with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. All run directories refuse overwrite. Publication is authorized. Repository: /home/loganriggs/Coding/reverse_engineering_llms, this subdirectory shared_tensor_networks.
