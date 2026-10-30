# Two vector blocks: exact alternating flow cannot improve the worst asymptotic quadratic rate

This extends TWO_BLOCK_FLOW_RATE.md from two scalar coordinates to two arbitrary coordinate blocks. It is an application of standard matrix inequalities, not a novelty claim or a theorem about global deep-network optimization.

## Statement and clocks

Let L(x)=x^T H x/2 with H symmetric positive definite. Let P be an orthogonal coordinate-block projector and Q=I-P. Each exact frozen-block flow runs for time tau>0. Its complete cycle is

C = exp(-tau QH) exp(-tau PH).

Joint flow with each coordinate active for the same time is J=exp(-tau H). The alternating physical duration is 2tau; this comparison does not equalize wall time or solver work.

**Proposition.** All eigenvalues of C are positive, its spectral radius is less than one, and

rho(C) >= exp(-tau lambda_min(H)) = rho(J).

Thus two-block exact alternation has no better asymptotic worst-case parameter contraction than joint flow under these clocks. A dominant eigenvector has per-cycle loss contraction rho(C)^2. The statement does not order losses for every initial state at finite time. Block reversal has the same eigenvalues.

## Proof

Set y=H^(1/2)x, U=H^(1/2) P H^(1/2), and V=H^(1/2) Q H^(1/2). Both U and V are positive semidefinite and U+V=H. The cycle in these coordinates is exp(-tau V)exp(-tau U). It is similar to the symmetric positive-definite matrix

T=exp(-tau V/2) exp(-tau U) exp(-tau V/2).

Let r=lambda_max(T)=rho(C). Since T <= rI in Loewner order, congruence by exp(tau V/2) gives

exp(-tau U) <= r exp(tau V).

The matrix logarithm preserves the order of positive-definite matrices. Therefore

-tau U <= log(r) I + tau V,

so -tau H <= log(r) I. Taking the largest eigenvalue proves log(r)>=-tau lambda_min(H).

For completeness, monotonicity of the logarithm follows from the convergent integral

log(A)=integral_0^infinity [(1/(1+s))I - (A+sI)^(-1)] ds.

If 0<A<=B, then (A+sI)^(-1)>=(B+sI)^(-1); integrating the difference gives log(A)<=log(B). This proof does not assume that the matrix exponential preserves Loewner order, which is false in general.

Finally, r<1: exp(-tau U) and exp(-tau V/2) are Euclidean contractions. If a nonzero y attained y^T T y=||y||^2, both contraction inequalities would be equalities, requiring y in ker(V) and ker(U). But ker(U) intersect ker(V)={0}, since U+V=H>0. Compactness of the unit sphere gives lambda_max(T)<1. Positivity follows from T>0. Reversing the two invertible factors is a similarity transformation, establishing equal spectra for reverse order.

## Strictness, symmetric splitting, and unequal durations

The scalar result was strict whenever the cross-coupling was nonzero. That assertion does not extend to general vector blocks. For example, let H=diag(.1, [[2,1],[1,2]]) in coordinates (x0,x1,x2), with P selecting (x0,x1). The coupled fast pair is strictly slower under alternation than joint flow within that pair, but the uncoupled x0 direction dominates both overall rates at tau=1. Both spectral radii equal exp(-.1), despite nonzero inter-block coupling.

Symmetric splitting with half a Q step, a full P step, and half a Q step is exactly T in energy coordinates. It has the same eigenvalues as the ordinary alternating cycle. It can improve discretization accuracy and finite trajectories without improving this asymptotic spectral rate at fixed tau. Since T is symmetric, its worst one-cycle loss ratio over initial states is exactly r^2; joint flow's is exp(-2tau lambda_min(H)). This is a worst-state comparison, not pointwise loss dominance.

If P and Q receive durations alpha and beta, respectively, the same proof compares their cycle with exp(-(alpha P+beta Q)H). Indeed the transformed generators sum to alpha U+beta V. Unequal block clocks therefore require comparison to the correspondingly preconditioned joint flow, not to unmodified joint flow at an arbitrary time.

## Application limits

For a nondegenerate local minimum in fixed coordinates, H is the local quadratic model. Bilinear models have scale gauges and sometimes additional nonidentifiability; a positive-definite chart must be justified before applying this proposition. Changing to a quotient chart may also change the block projectors. This note does not assert that every layerwise neural-network update satisfies these assumptions.

The proof covers exactly two complementary blocks and exact flows on a fixed SPD quadratic. It does not establish an ordering for three or more blocks, state-dependent Hessians, indefinite curvature, stochastic gradients, adaptive optimizers, exact block minimization, or generalization/recovery. A fixed positive-definite parameter metric can be whitened when its transformed block updates satisfy the same projector assumptions; arbitrary independent block preconditioners require their own specification.

The inequality is closely related to the established Araki-Lieb-Thirring exponential/product comparisons. For primary mathematical context see K. M. R. Audenaert, [An Araki-Lieb-Thirring inequality for geometrically concave and geometrically convex functions](https://arxiv.org/abs/1204.3418). The self-contained logarithm argument above suffices for the largest-eigenvalue comparison needed here; no stronger log-majorization theorem is claimed or needed.

## Reproducible checks

Run check_vector_block_flow_rate.py with single-threaded NumPy/SciPy on CPU. It compares literal frozen-block affine solution maps against matrix exponentials and the transformed symmetric product, checks reversed and symmetric-split spectra, verifies selected switched ODE endpoints, and includes uncoupled and coupled-but-equal-rate controls. Results and source hashes are written to a fresh results/vector_block_flow_rate directory. Numerical fixtures test implementation; the proof supplies the general inequality.
