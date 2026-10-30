# Exact two-block flow: a local limit on what alternation can improve

This is an elementary two-dimensional result, not a novelty claim or a theorem about arbitrary deep-network optimization. It gives a concrete local comparison for direction B, alongside the previously completed nonlinear alternation experiments.

Consider L(x,y)=1/2(ax²+2cxy+by²), with a,b>0 and c²<ab. The Hessian H is positive definite; gauge directions must already have been removed for such a local model to apply. Compare joint Euclidean gradient flow for time tau with an exact x-only flow for time tau followed by an exact y-only flow for time tau. These clocks give each coordinate equal active time. The alternating physical duration is2tau, and no equality of numerical solver cost is assumed.

Let u=exp(-a tau), v=exp(-b tau), q=c²/(ab). The exact block maps are

A=[[u,-(c/a)(1-u)],[0,1]],
B=[[1,0],[-(c/b)(1-v),v]],
C=B A.

Their cycle determinant is uv and their trace is u+v+q(1-u)(1-v). Both eigenvalues are positive; since0<=q<1, the trace is less than1+uv, and the larger eigenvalue is less than1. Thus alternating flow converges in this model.

**Proposition.** The asymptotic worst-case parameter contraction per alternating cycle satisfies

rho(C) >= exp(-lambda_min(H) tau).

The right side is the corresponding contraction for joint gradient flow. The inequality is strict when c is nonzero and tau>0. Therefore finite-dwell alternation cannot improve this asymptotic rate in the specified two-scalar-block positive-definite quadratic model, even when clocks are matched by each coordinate's active time.

## Proof

Put S=a+b and D=a-b. Multiplying both traces by exp(S tau/2), the alternating trace becomes

2[(1-q)cosh(D tau/2)+q cosh(S tau/2)],

while the joint-flow trace becomes

2cosh(sqrt(D²+4c²) tau/2).

The identity D²+4c²=(1-q)D²+qS² reduces the trace comparison to Jensen's inequality for g(t)=cosh(sqrt(t) tau/2), t>=0. Its power series has nonnegative coefficients and a strictly positive quadratic coefficient for tau>0, so g is strictly convex. Hence trace(C)>=trace(exp(-tau H)), strictly for0<q<1 because S²-D²=4ab>0.

The two maps have the same positive determinant exp(-S tau), and both have positive real eigenvalues. For fixed determinant d, the larger eigenvalue (T+sqrt(T²-4d))/2 is increasing in trace T. The claimed spectral-radius comparison follows. At c=0 the two block maps commute and the complete cycle equals joint flow.

The eigenvalues of C are real because its discriminant equals (u-v)²+2q(u+v)(1-u)(1-v)+q²(1-u)²(1-v)², which is nonnegative. Positive trace and determinant then give positivity. The trace bound T<1+d gives the strict upper bound rho(C)<1. These steps avoid inferring an eigenvalue ordering from trace alone in higher dimensions.

## What the result does and does not explain

This is an asymptotic spectral-rate comparison, not a pointwise ordering of loss for every initial vector or every finite iteration. Loss on a dominant eigenvector decays by rho(C)² per cycle. Reverse order has the same eigenvalues (A and B are invertible), but generally a different finite trajectory.

The earlier nonlinear four-factor test found that finite switching can leave a symmetry-preserving trajectory. That mechanism lies outside this positive-definite quadratic comparison. So can large nonconvex excursions, stochastic effects, adaptive metrics, unequal clocks, vector-valued blocks and exact block minimization. The proposition does not rank Adam, Gauss–Newton, wall time, generalization or circuit recovery.

For our research direction, ordinary cyclic switching has no automatic local acceleration rationale merely because blocks are coupled. A claimed gain should identify the additional mechanism: improved conditioning, a different metric, a finite nonconvex escape, or cheaper computation. This complements the matched-gradient-work negative result for layerwise alternation; it does not explain that entire experiment.

Implementation checks are in check_two_block_flow_rate.py and results/two_block_flow_rate. They compare the closed block maps to independent matrix exponentials, the rate inequality over varied curvature and coupling, and selected switched trajectories against numerical ODE integration. The proof, not a finite test grid, supplies the general statement.
