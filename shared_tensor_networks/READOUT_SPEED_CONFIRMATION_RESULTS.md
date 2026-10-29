# Fresh-teacher readout-speed confirmation

All 70 predeclared fits are terminal; 70/70 reached the target accumulated hidden-step time of 20. Five fresh teachers and two fresh initializations per teacher. This panel is separate from the previous archived-teacher results. Same seven methods, fixed rates and line-search settings; no rescue or tuning. See READOUT_SPEED_CONFIRMATION_PLAN.md.

Each error is a median of five teacher means, averaging both starts before taking the median. Transfer also averages four whole-task holdouts, each adapted with 128 noisy labels. Recovery requires all four matched quadratic features to exceed absolute Gaussian cosine .99. Failed or capped cases are retained in all aggregates.

| Method | Readout speed | Source MSE | Transfer MSE | All features recovered | Clock reached | Median gradient calls |
|---|---:|---:|---:|---:|---:|---:|
| joint | 1 | 0.0318666 | 0.0912673 | 1/10 | 10/10 | 2000 |
| joint | 10 | 0.0819402 | 0.246752 | 1/10 | 10/10 | 5279 |
| joint | 100 | 0.0823329 | 0.195374 | 0/10 | 10/10 | 44507 |
| reduced | exact optimum | 0.0769562 | 0.190597 | 0/10 | 10/10 | 2000 |
| split | 1 | 0.0316748 | 0.0909116 | 1/10 | 10/10 | 2000 |
| split | 10 | 0.0818351 | 0.251717 | 1/10 | 10/10 | 2000 |
| split | 100 | 0.0793198 | 0.193915 | 0/10 | 10/10 | 2000 |

## Paired comparisons

- Split speed 10 has lower teacher-mean transfer than ('joint', 1) on 2/5 fresh teachers.
- Split speed 10 has lower teacher-mean transfer than ('joint', 10) on 2/5 fresh teachers.
- Split speed 10 has lower teacher-mean transfer than ('joint', 100) on 0/5 fresh teachers.
- Split speed 10 has lower teacher-mean transfer than ('reduced', 0) on 1/5 fresh teachers.
- Split speed 10 has lower teacher-mean transfer than ('split', 100) on 1/5 fresh teachers.

The earlier moderate-speed advantage did not replicate: split10 beats joint1 on only 2/5 fresh teachers and reduced on only 1/5, with worse median transfer than both. Joint1, split10 and reduced all had hidden clock20 in BOTH panels, so this primary reversal is not explained by the new clock matching. Teachers and initializations both changed; this experiment does not isolate which caused the reversal.

At a common speed, the aggregate joint and split errors are similar here, while split updates use fewer gradients at high speed. This suggests a numerical integration benefit for the stiff readout block, not a demonstrated benefit for recovering reusable computation. Single-run timings and different matrix-solve costs prevent a general compute-efficiency claim.

These counts and the aggregate medians do not constitute a universal method ordering. The five teachers are the independent experimental blocks; the two initializations and four holdouts are not additional independent teacher replications. The predeclared primary comparisons are split10 versus joint1 and reduced.

## What the clock control does and does not remove

Matching the sum of accepted hidden step sizes removes the earlier gross reduction in hidden-learning time when fast explicit updates backtracked. It does not equalize compute, wall time, discretization error or physical time in a switched ODE. Joint methods update both blocks explicitly; split methods retract hidden factors then integrate only the frozen-hidden readout exactly. Reduced methods solve the readout at each candidate state. These are finite-step algorithms, not error-controlled full ODE integrations.

Per-fit elapsed CPU time, gradient and forward evaluations, readout solves, exact block-flow calls, attained clocks and all statuses are retained. Timing is one run on one pinned CPU core, not a robust hardware benchmark. The target clock is a finite budget, not a convergence certificate.

## Verification

Independent audit checks 70 checkpoints and 32556 stored states, fresh source/holdout target and noise generation, teacher Gaussian RMS, source-data hashes, common initial states, gradient/resource counters and the last shortened step. Literal-forward objective replay, SVD adaptation, Wick source/transfer moments, and quadratic coefficient matching pass. All final gradients agree with independent Torch autograd; maximum absolute error 5.45e-15. All internal line-search decisions are not independently replayed.

## Decision

Close this bounded fresh-teacher/clock control; do not expand a rate search on these teachers. Preserve the full method table rather than promoting the best aggregate as a universal optimizer. The next broad-goal experiment is the outstanding d64/h32 task/input scaling grid, retaining both data coverage and optimization diagnostics. Native MLP research remains paused; the overall toy-research goal remains active.
