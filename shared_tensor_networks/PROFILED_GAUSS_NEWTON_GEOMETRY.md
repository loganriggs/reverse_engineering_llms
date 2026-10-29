# Exact readout elimination is not joint Gauss–Newton with a smaller solve

For one output (sum over outputs for multiple heads), let r(theta)=Z(theta)u*(theta)-y, with u* the unique least-squares readout. Z can be the augmented design [Phi;sqrt(lambda)I]. Assume full column rank in a neighborhood. Let P=Z(ZᵀZ)^{-1}Zᵀ. Readout optimality gives Zᵀr=0. Differentiating yields

 dr = (I-P)(dZ)u* - Z(ZᵀZ)^{-1}(dZ)ᵀr.

Define A by its action A dtheta=(I-P)(dZ)u*, and B dtheta=−Z(ZᵀZ)^{-1}(dZ)ᵀr. Then A lies orthogonal to col(Z), while B lies in col(Z). Therefore

 J_reduced=A+B,
 AᵀB=0,
 J_reducedᵀr=Aᵀr,
 J_reducedᵀJ_reduced=AᵀA+BᵀB.

In contrast, joint Gauss–Newton linearizes with independent hidden and readout increments. Minimizing its linear residual over the readout increment leaves r+A dtheta. Its profiled curvature is AᵀA. The full reduced-residual GN curvature includes the extra positive-semidefinite BᵀB, despite the SAME gradient. B vanishes at zero augmented residual, but need not vanish away from a fit (including with readout ridge).

For a radial penalty independent of the readout, append its residual/Jacobian to both problems. Those rows contribute identically to A and zero to B, so the identities continue to hold. For multihead vectorization, use the column space of the full readout Jacobian; the same projection argument applies.

These are exact local algebraic identities, not claims of new variable-projection theory or superior convergence. They do not show that this extra term causes the observed basin failures. Adaptive TRF trust-region radii and scaling introduce additional differences in our existing runs.

## Numerical control on actual networks

check_profiled_gauss_newton.py evaluates five archived d8/h4 normalized starts, each at ridge0/.001 (ten fixtures). It independently constructs the joint and exact reduced residual Jacobians, projects the hidden joint Jacobian off the readout-Jacobian column space, and checks orthogonality, gradient equality and the curvature identity. It also solves the full joint linear system with damping .001 ONLY on hidden coordinates and compares its hidden step to the projected solve. This damping convention is necessary: damping readout increments would change the elimination formula.

All ten checks pass. Extra-curvature trace tr(BᵀB)/tr(AᵀA) ranges approximately .479–.781 at these initializations. This measures aggregate curvature, not a bound on every eigenvalue or a predicted convergence slowdown. Details and numerical residuals are in results/crossed_readout/profiled_gn_checks.json.

## Next discriminating experiment

At each state fit the readout exactly, then use either AᵀA+lambda_step I or AᵀA+BᵀB+lambda_step I for the hidden step. Keep the actual reduced objective, normalization/radial control, initial state, fixed step damping and Armijo search identical. Record accepted loss decrease, step sizes, linearization error, component recovery and whole-task transfer. This isolates a curvature-model distinction without varying source labels or the objective. It is not exact alternating least squares over hidden factors and should not be called so.
