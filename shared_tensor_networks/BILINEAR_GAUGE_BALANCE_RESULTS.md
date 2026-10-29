# Function-preserving factor balancing

54 fixed-damping Gauss–Newton fits: 18 archived teacher/student/metric fixtures times original, initial-only balancing, and balancing initially plus after every accepted step. Same initial function, frozen normalization, damping .001, Armijo schedule, 100-Jacobian cap and MSE<1e-12 stopping criterion. No restarts or damping tuning. Balance paired rows A/B, AR/BR when present, and C/D by reciprocal positive scaling. Zero-norm pairs are left unchanged. The readout is unchanged.

| Student | Balance schedule | Median population MSE | Below 1e-12 | Median Jacobian calls |
|---|---|---:|---:|---:|
| tied_raw | original | 2.09442e-05 | 1/6 | 100 |
| tied_raw | initial | 5.49438e-08 | 2/6 | 100 |
| tied_raw | every | 4.86706e-07 | 1/6 | 100 |
| tied_corrected | original | 0.000438354 | 0/6 | 100 |
| tied_corrected | initial | 4.27039e-07 | 1/6 | 100 |
| tied_corrected | every | 2.43995e-05 | 1/6 | 100 |
| untied_raw | original | 6.31872e-13 | 6/6 | 17 |
| untied_raw | initial | 6.17752e-13 | 5/6 | 17 |
| untied_raw | every | 2.42418e-13 | 6/6 | 13 |

initial: 9 lower-error outcomes, 6 tolerance ties, 3 higher-error outcomes versus original coordinates across 18 paired fixtures.

every: 8 lower-error outcomes, 6 tolerance ties, 4 higher-error outcomes versus original coordinates across 18 paired fixtures.

At initialization, balancing improves the condition number of the 15 nonzero weighted-Jacobian singular directions in 12/18 fixtures. These include repeated initial parameterizations across teacher families and metrics, so they are not 18 independent draws. Ratios and individual outcomes are saved in summary.json. This diagnostic alone does not explain final performance.

One failure is retained: untied teacher seed211003, raw tied student, every-step balancing exhausted all 12 line-search candidates at iteration98, MSE1.9970e-5. The independent audit reproduces the descent direction and rejection of every candidate. No retry was performed. Other fits remain finite; not every successful run reaches stopping tolerance.

## What balancing does mathematically

For nonzero paired vectors a,b, set c=sqrt(||b||/||a||). Then (ca)(b/c)^T=ab^T and both new norms equal sqrt(||a||||b||). This minimizes c²||a||²+c⁻²||b||² along the positive reciprocal-scaling orbit, by AM–GM or differentiation. It fixes this particular scale freedom, not all network symmetries. Equal parameter norm does not guarantee an optimally conditioned output Jacobian.

Keeping the same numerical damping matrix after rescaling deliberately changes the optimizer metric. If theta=T phi is a fixed invertible change of coordinates, transporting damping to TᵀMT would preserve the corresponding linearized update; using I or the same multiplicity matrix in the new coordinates does not. Thus these are algorithmic interventions on geometry, not an assertion that Gauss–Newton is coordinate invariant. Rebalancing after each step is state dependent.

## Verification and decision

All 18 original trajectories reproduce archived GN states exactly. Every rebalanced state has independently matching polynomial coefficients and paired outer products before/after balancing; applicable row norms match. All trajectory losses replay through exact Gaussian coefficient moments. Audits verify raw pre-balance step equations, Armijo descent, sampled autograd Jacobians and the failed final search. Both raw and balanced states are retained.

Initial-only balancing improves the tied-model median errors, but neither schedule universally wins. This supports sensitivity to function-preserving coordinates and does not prove that reciprocal row imbalance accounts for all difficult trajectories. These are tiny population-regression tasks, not sample-efficiency or circuit-recovery results. Close the bounded geometry controls; next prioritize multiple-readout/intervention supervision and held-out transfer to address hidden-computation recovery, rather than expanding optimizer rescue sweeps.
