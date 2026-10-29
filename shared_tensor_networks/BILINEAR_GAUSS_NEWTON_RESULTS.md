# Damped Gauss–Newton on the bilinear toy tasks

36fits reuse the exact teacher, initial function, frozen normalization and Gaussian population objective from the layerwise-alternation panel. Two fixed damping values .001/.1; at most100linear solves; Armijo backtracking over twelve powers of1/2; stop at MSE<1e-12. No damping adaptation, restarts or best-run selection. Raw damping usesI; corrected tied damping uses occurrence metricM (2on bottomA/B,1elsewhere).

| Teacher | Student / damping metric | Damping | Median population MSE | Below1e-12 |
|---|---|---:|---:|---:|
| tied | tied_raw | 0.001 | 4.17622e-05 | 0/3 |
| tied | tied_raw | 0.1 | 0.000782296 | 0/3 |
| tied | tied_corrected | 0.001 | 3.75092e-05 | 0/3 |
| tied | tied_corrected | 0.1 | 0.000788505 | 0/3 |
| tied | untied_raw | 0.001 | 7.12313e-13 | 3/3 |
| tied | untied_raw | 0.1 | 8.25002e-06 | 0/3 |
| untied | tied_raw | 0.001 | 2.49465e-10 | 1/3 |
| untied | tied_raw | 0.1 | 0.00185702 | 0/3 |
| untied | tied_corrected | 0.001 | 0.00954277 | 0/3 |
| untied | tied_corrected | 0.1 | 0.00213279 | 0/3 |
| untied | untied_raw | 0.001 | 5.98957e-13 | 3/3 |
| untied | untied_raw | 0.1 | 1.25898e-07 | 1/3 |

All36endpoints improve on their matching joint-gradient baseline (step.003,10000updates), but this is NOT a matched-compute optimizer tournament: Gauss–Newton forms full Jacobians, solves dense parameter-space systems and performs line searches. Counts and wall times are saved. No line search exhausted; maximum accepted backtracking trial count6.

For untied students with damping.001, all six teachers reach MSE<1e-12 in7–71steps. Tied students remain sensitive to metric/damping and initialization; some retain error near.074. Thus finite-budget gradient error should not be treated as a representational floor. Full-model factor/circuit recovery has not been established.

Independent verification: every accepted trajectory state is evaluated by monomial expansion and exact Gaussian coefficient moments; all steps satisfy the linear-system and Armijo conditions. Initial/middle/final analytic Jacobians match torch.func.jacrev. Maximum relative Jacobian error7.31e-16; maximum normalized linear residual2.55e-10; maximum coefficient-moment loss discrepancy1.45e-14. All36fits finite.

Tangent-space diagnostic: both tied and untied initial Jacobians have numerical rank15, the dimension of homogeneous quartics in three variables. Paired output tangent projectors agree; undamped linearized fitting can remove the residual at the starting point numerically. The smallest nonzero singular value is only5.31e-6–2.88e-4of the largest. In these tiny models, the observed difference therefore cannot be described simply as the untied model having additional first-order output directions at initialization. Conditioning, damping, finite steps and nonlinear paths remain relevant. Rank is a floating-point diagnostic, not an exact symbolic certificate or a guarantee of global representability.

Conditional mathematical statement: on a probe set with injective evaluation of homogeneous quartics, any such model whose Jacobian has rank15 spans the full quartic evaluation space locally. If both tied and untied Jacobians have that rank, their output tangent spaces coincide. This follows because both are subspaces of the same15dimensional polynomial evaluation space. It does not equate their parameter metrics or finite nonlinear updates.

Next: keep this optimizer result bounded; do not tune damping until every tied run succeeds. Use the geometry controls to design a separately budgeted curvature-aware comparison, while completing the five-seed image data/capacity confirmation.
