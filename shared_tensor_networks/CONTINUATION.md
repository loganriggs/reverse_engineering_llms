# Toy research continuation — active, 2026-10-29

Goal: complete the toy experiments, scale them as justified, and derive/test interesting theoretical results. The prior native-MLP research remains paused. This project is separate and must not be redirected automatically into the old featurizer loop.

Previous substantive goal work: progress (implemented E0/E1/E2 controls). Current continuation: progress (new regression controls, d=8 learning study, independent audits, and algebraic characterization). No blocker and no live experiment left at this checkpoint.

## Evidence current through this batch

- Initial: README.md, MATHEMATICS.md, RESULTS.md; E0, d=4 E1, E2 metric control.
- New: REGRESSION_AND_D8_RESULTS.md and POLYNOMIAL_QUOTIENT_GEOMETRY.md.
- Frozen data/results: results/e1_regression (640 records), results/e1_d8 (150 fits), results/e1_d8_regression (400 records), results/multiplication_geometry.
- All above producers and auditors are terminal and passed. Do not rerun or overwrite their output directories.
- Same source experiment.py as the original harness; d=8 uses h=q=8, lr=.003, matched initialization, 600 steps, five seeds, N=64,128,256,512,1024, both teacher types. Not an exact replication of the user's pasted script; differences are documented.

## Findings that change the next action

Quartic least squares outperforms the neural fits and recovers all targets once N exceeds the quartic coefficient count. Oracle hidden features recover d=8 targets already at N=64. Need to isolate learning useful features versus fitting the top readout. No general tying advantage, circuit recovery, or basis-invariant cancellation interpretation established.

Polynomial identities induced by repeated inputs create a multiplication-map kernel. For fixed symmetric Q, minimum bilinear top width is max(positive inertia, negative inertia). For a fixed hidden polynomial dictionary, minimize that over all equivalent Q. Verified constructive identities; no novelty claim. At d=4 untied hidden maps span the entire quartic space, so that setting is weak for distinctive feature recovery. d=8 oracle maps are substantially lower dimensional (36/64 versus330).

## Next bounded experiment (not yet started)

Refit the top scalar readout and the full hidden-product readout of ALL frozen d=8 trained states, alongside exactly matched random-initialized and oracle controls. Use existing sample sets, both teachers, all five seeds and sizes. Freeze the ridge grid and metric summaries before outcomes; use validation only. Separate changes to top representable class from optimizer-only changes. Save maps/coefficient outputs; audit direct forward equivalence, exact population errors, and solve residuals. No seed/example selection or extra learning-rate tuning. This is a diagnostic on reused development data, not fresh confirmation.

## Remaining scope — not complete

- E1: independent teacher/initialization factorial confirmation if an interpretable advantage emerges; original-script corrected setting is only partially matched, not a literal replication.
- E2: additional same-function geometry controls, learning-budget versus convergence distinctions, and useful function-subspace recovery metrics. Existing exact projected-SGD equivalence is complete, not an optimization theorem.
- E3: an explicit identifiable graph family or observable equivalence class, controlled parameter-only/input/intermediate reuse, graph complexity penalties, and intervention/held-out-composition tests. Exact function recovery alone is insufficient.
- Theory: small fiber/critical-locus calculations and explicit likelihood-based singularity work can extend the current notes. Do not infer RLCT from Hessians or claim literature novelty without a source review.
- E4: larger input/width/depth CPU→GPU runs when informative. Respect existing GPU workloads; do not scale just to spend compute or treat loss failures as structural results.
- Track sample error, population function error, parameter count, arithmetic execution cost, and graph recovery separately. The entire requested research program is not achieved.

Runtime: /home/loganriggs/Coding/neuron_circuits/.venv/bin/python with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. All run directories refuse overwrite. Publication is authorized. Repository: /home/loganriggs/Coding/reverse_engineering_llms, this subdirectory shared_tensor_networks.
