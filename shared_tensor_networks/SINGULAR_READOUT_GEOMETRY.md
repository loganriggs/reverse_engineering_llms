# Rank, usable capacity, and singular readouts

Direct derivations and controlled tests, with no claim of literature novelty. This note concerns a fixed-feature linear readout and parameter interpolation; it does not estimate a network RLCT or prove a particular neural training mechanism.

## Exact error decomposition

Let φ(x) be a fixed feature vector, and let S be its linear span in the Gaussian L² function space. Let p be the target and p_S its orthogonal projection onto S. For any readout f in S:

`E[(f−p)²] = E[(p_S−p)²] + E[(f−p_S)²]`.

The cross term vanishes because p−p_S is orthogonal to S. The first term is a representation floor for that fixed linear readout class. The second contains finite-data estimation, regularization, and readout fitting error. This is not a unique decomposition of end-to-end neural optimization: changing the hidden representation changes S, and allowing all hidden products enlarges the original factored top class.

For quartic coefficients c and moment matrix H, use white coordinates H^(1/2)c, implemented as Lᵀc for H=LLᵀ. If K maps readout weights to polynomial coefficients, S is represented by the columns of LᵀK. The saved refit audit verifies the Pythagorean identity for all 1,400 fitted readouts to scaled error below 3.56e-15.

## An exact singular feature example

Take independent standard-normal scalar features u,v and expose `(u, u+εv)` to a linear readout. The target is v.

- At ε=0, the feature span contains only u; the minimum population MSE is 1.
- At every ε≠0, v is exactly represented by weights `(-1/ε,1/ε)`, so the minimum population MSE is 0.
- The required weight norm is √2/|ε|. Thus exact linear-span capacity jumps discontinuously at ε=0 while the feature map itself changes continuously.

This is not a discontinuity of any bounded-weight predictor. With a fixed coefficient norm bound, the prediction's v component vanishes as ε→0. A fixed positive ridge penalty likewise yields continuous predictions and error approaching 1. The unbounded optimum allows weights to diverge.

For population half-squared loss

`L(a,b)=0.5[(a+b)²+(εb−1)²]`,

the readout Hessian is `[[1,1],[1,1+ε²]]`. Its smallest eigenvalue is

`λ_min = 2ε² / (2+ε²+sqrt(4+ε⁴)) ≈ ε²/2`.

Plain gradient descent along the slow direction contracts by `1−ηλ_min` per step (for a stable fixed step size η), requiring a characteristic O(ε⁻²) number of steps. Feature-span rank does not encode this optimization cost. The script verifies the Hessian with autograd and the exact eigenvalues at seven ε values. Hessian conditioning here is not an RLCT calculation.

## Robustness depends on the noise model

If each exposed feature receives independent zero-mean noise with variance σ², the exact-fit difference readout adds noise variance `2σ²/ε²`. A 200,000-draw Monte Carlo check verifies this prediction at ε=.01,.1,1. The same noise draws are reused across ε; these are scaling checks, not independent replications.

If the noise is identical in both features instead, the readout cancels it exactly. Input noise that propagates jointly through both features is yet another model. Thus “redundant,” “cancelling,” and “robust” cannot be assigned without saying where perturbations enter and how they correlate.

## Measured in the bilinear-network toy family

We take all 50 width-matched untied d=8 states from the original 600-step study and interpolate their hidden parameters from their equal-branch initializations to their trained values. These are straight parameter paths, not recorded training trajectories. Each hidden-product feature is normalized to unit Gaussian RMS before comparing readout conditioning. This removes simple per-column scale differences, not correlations or general basis dependence.

At ε=0 every hidden-product map has rank 36. At ε=10⁻⁵ every map has rank 64 under the frozen relative singular-value cutoff 1e-10. Across these paths:

| Interpolation ε | Median unregularized population floor | Median readout norm | Median population error with ridge λ=.001 |
|---:|---:|---:|---:|
| 0 | 0.806 | 0.485 | 0.806 |
| 0.00001 | 0.580 | 76,379 | 0.806 |
| 0.001 | 0.579 | 764 | 0.805 |
| 0.01 | 0.579 | 76.4 | 0.774 |
| 0.1 | 0.565 | 7.86 | 0.583 |
| 1 | 0.295 | 1.16 | 0.295 |

This mirrors the simple mechanism: a large gain in ideal span capacity can appear before a bounded, regularized readout can exploit it. These pooled medians reuse teacher/seed initializations across sample-size paths; they are descriptive, not 50 independent confirmations. The ridge uses the true population target as a diagnostic and is not a trainable predictor in our finite-data comparisons.

The full projection floor remains mathematically meaningful, but must be accompanied by singular values and required coefficient norms. Its improvement alone is too weak to establish stable or learnable reusable structure.
