# Intervention design survives matched-prior regularization

Fresh-seed confirmation completed:20 new teachers/pools/noise/test sets,1280 endpoints, all predeclared settings retained. Source hashes, full records and numerical checks are in results/regularized_joint_design/results.json. This is a known quadratic contrast family, not discovery of unknown circuitry.

At28 measurements, median clean held-out intervention MSE:

| Noise SD | Estimator | Random | Pair-balanced | Designed |
|---|---|---:|---:|---:|
| .05 | Minimum-norm LS | .09560 | .03943 | .00260 |
| .05 | Matched ridge | .08500 | .01751 | .00253 |
| .2 | Minimum-norm LS | .23728 | .63087 | .04162 |
| .2 | Matched ridge | .11819 | .07796 | .03304 |

Designed probes beat both ridge baselines20/20 paired seeds at both noise levels at budgets28 and56. At budget14 the corresponding wins are16/20 versus random and15/20 versus balanced for low noise;20/20 and18/20 for high noise. At budget112 they are20/20 against both at low noise and19/20 against both at high noise. There is no universal individual-teacher dominance claim. All strategies use the same unit-norm candidate pool; singleton measurements remain uninformative by construction.

Regularization removes much of the unstable-estimator penalty. The residual advantage supports a narrower claim: deliberate measurement coverage and strength remain valuable even with the correct prior and noise-aware estimator. The supplied teacher prior is N(0,I); ridge penalty sigma^2 is its posterior-mean rule. This privileged prior must not be presented as something learned from an unknown network.

## Analytic risk check

For y=A theta+epsilon, theta~N(0,I), epsilon~N(0,sigma^2 I), let B=(A^T A+sigma^2 I)^-1. The estimator is B A^T y and its integrated error covariance is

    (B A^T A-I)(B A^T A-I)^T + sigma^2 B A^T A B
      = sigma^2 B = C.

The acquisition design is independent of theta and the observation noise, so conditioning on that design does not alter the prior used here. For a uniformly chosen support of size k and uniform spherical amplitudes, distinct-coordinate squared products have mean (k-1)/(8*7*(k+2)). Sign symmetry makes off-diagonal monomial moments zero. Averaging equally over k=3,4 gives test second-moment matrix cI, c=.008035714285714285. Thus exact prior/noise/test-averaged prediction risk conditional on a design is c*trace(C). This is ordinary Gaussian linear estimation, not a new general theory.

At28 measurements, averaging that exact risk over the20 candidate designs gives:

| Noise SD | Random | Balanced | Designed | Singleton |
|---|---:|---:|---:|---:|
| .05 | .09003 | .02250 | .002385 | .225 |
| .2 | .12369 | .08502 | .03282 | .225 |

These are conditional Bayes risks averaged over designs, not medians of realized teacher errors. They independently explain the empirical ordering without asserting equality between the two summaries. The identity can also be written c*sum sigma^2/(I_j+sigma^2) because each acquisition measures just one pair coefficient. Missing information contributes a full prior variance; weak measurements contribute a large posterior variance.

Ridge matched augmented least squares within2.23e-15 and independent scalar solutions within8.89e-16. The bias-plus-noise covariance matched posterior covariance within2.23e-16. Checked support-size holdouts, perturbation norms, unique candidates and all1280 endpoints. The three implementations share generated inputs; this is numerical/algebraic validation, not an independent replay from separate data generators.

## Decision

Close this known-interface estimator comparison: no additional ridge tuning or optimizer sweeps. The important next difficulty is identifying an unknown feature basis or detecting a misspecified interaction family. Geometry-based design here uses known contrasts and a correct quadratic model. A useful extension must allow competing explanations to disagree and test held-out interventions beyond those used to construct them, rather than interpreting success in this positive control as real-model mechanistic understanding. Native-model work remains paused; h128 scaling remains active under its frozen protocol.
