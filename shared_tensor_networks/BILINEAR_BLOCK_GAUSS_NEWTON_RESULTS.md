# Alternating block Gauss–Newton on bilinear networks

All 54 fits independently audited. Same 18 archived teacher/student/metric fixtures as full Gauss–Newton, damping .001, Armijo search, maximum 100 full-Jacobian evaluations, stop at population MSE <1e-12. Methods: full, forward bottom/top/readout, reverse readout/top/bottom. Each alternating iteration updates one block, not an entire sweep. Full Jacobians and Gram matrices are computed in every method; block methods solve smaller systems. This is an equal Jacobian-budget cap, not equal FLOPs, wall time, or specialized block implementation. No new hyperparameter search.

| Student | Method | Median final MSE (six teachers) | Below 1e-12 | Median Jacobian calls |
|---|---|---:|---:|---:|
| tied_raw | full | 2.09442e-05 | 1/6 | 100 |
| tied_raw | forward | 0.0011277 | 0/6 | 100 |
| tied_raw | reverse | 0.000306878 | 0/6 | 100 |
| tied_corrected | full | 0.000438354 | 0/6 | 100 |
| tied_corrected | forward | 4.87731e-05 | 0/6 | 100 |
| tied_corrected | reverse | 0.00078505 | 0/6 | 100 |
| untied_raw | full | 6.31872e-13 | 6/6 | 17 |
| untied_raw | forward | 4.38642e-13 | 5/6 | 44.5 |
| untied_raw | reverse | 6.36423e-13 | 6/6 | 40.5 |

forward: 4 lower-error outcomes, 5 both below stopping tolerance, 9 higher-error outcomes versus full GN (18 paired fixtures).

reverse: 4 lower-error outcomes, 6 both below stopping tolerance, 8 higher-error outcomes versus full GN (18 paired fixtures).

Full-method parameter trajectories reproduce all 18 archived .001-damping runs exactly. Every accepted state passes independent polynomial-coefficient Gaussian-moment loss replay; selected-block normal equations and Armijo descent pass; unselected parameters are unchanged. Initial/middle/final analytic Jacobians match autograd. All fits finite; no line-search failures.

Interpretation: curvature-aware alternating updates can reach near-exact fits and improve some difficult tied fits, but do not show a universal advantage over full GN. Full GN solves all six untied-student cases; forward solves five, reverse six. Tiny differences below the stopping tolerance are not evidence of superiority. The timing totals are subsecond and single-run, so they are not reliable hardware performance benchmarks. These are function-fitting results, not circuit recovery.

## Why alternating curvature steps can differ

At a common state, let Q(delta)=rᵀJ delta + 1/2 deltaᵀ(JᵀJ+lambda M)delta, with lambda>0 and M positive definite. Full damped GN uniquely minimizes Q over all parameters. A block step minimizes the same Q over vectors supported on that block; consequently Q(full)<=Q(block)<=Q(0). Both nonzero steps are descent directions because g_Bᵀ delta_B=−delta_BᵀH_BB delta_B<0. This is a direct quadratic-minimization argument, not a nonlinear convergence theorem.

After a block moves, the next block sees a different Jacobian and residual. Smaller restricted steps may remain accurate farther along a nonlinear trajectory, and line-search acceptance may differ. Thus local model dominance does not order final nonlinear errors. This identifies the relevant extension: measure linearization error and accepted step sizes before attributing block benefits to missing output directions.

Decision: close this bounded block-order comparison. Next use a function-preserving gauge-balancing control to test conditioning directly, or multiple-readout supervision to address the separately demonstrated hidden-recovery failure. Do not expand optimizer sweeps merely to make every tied fit succeed.
