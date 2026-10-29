# Alternating least squares: optimizer control and remaining traps

All54four-factor ALS fits completed on the exact data, noise, initial factors and frozen normalization from the finite-data SGD study.100sweeps, fixedfactororder0/1/2/3; each sweep solves four3-parameter linear least-squares problems. This is400blocksolves versus1000SGDsteps, not a compute-matched performance comparison. No restarts, regularization, gauge changes or best-run selection.

For factor i, let b_n=product_(j!=i)(a_j dot x_n)/scale. With other factors fixed, prediction is b_n*x_n^T*a_i, so solve designD_(n,:)=b_n*x_n^T against labels. The subproblem is a convex quadratic. A global block solve cannot increase training loss; a full-rank design gives a unique block minimum. Repeated/shared factors enter with higher powers, so this linear update does not apply directly to those shared blocks.

Independent audit checks all21,600blockupdates: unchanged other factors, normal equations, training-loss monotonicity and objective replay. Maximum normalized normal-equation residual4.534e-15. Final population error from analytic coefficient moments independently matches Gaussian quadrature for all54fits. Original data/init/result hashes are retained.

## Exact population MSE, medians over3seeds

| Teacher factors | Noise SD | N | Four-factor SGD | Four-factor ALS | ALS strict recovery count |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 32 | 0.90692 | 0.0013686 | 1/3 |
| 1 | 0 | 128 | 0.04276 | 0.32301 | 1/3 |
| 1 | 0 | 512 | 0.016665 | 1.1934e-05 | 3/3 |
| 1 | 0.1 | 32 | 0.92969 | 1.5737 | 0/3 |
| 1 | 0.1 | 128 | 0.038194 | 0.32891 | 1/3 |
| 1 | 0.1 | 512 | 0.015915 | 0.00010435 | 2/3 |
| 2 | 0 | 32 | 0.8124 | 0.00031249 | 1/3 |
| 2 | 0 | 128 | 0.070149 | 8.2346e-06 | 3/3 |
| 2 | 0 | 512 | 0.023937 | 1.5048e-05 | 3/3 |
| 2 | 0.1 | 32 | 0.81051 | 0.10193 | 0/3 |
| 2 | 0.1 | 128 | 0.074871 | 0.0014547 | 1/3 |
| 2 | 0.1 | 512 | 0.023647 | 0.00023182 | 2/3 |
| 4 | 0 | 32 | 0.79869 | 0.00019139 | 2/3 |
| 4 | 0 | 128 | 0.18701 | 8.5896e-08 | 3/3 |
| 4 | 0 | 512 | 0.046721 | 3.4311e-30 | 3/3 |
| 4 | 0.1 | 32 | 0.78084 | 0.084483 | 0/3 |
| 4 | 0.1 | 128 | 0.18701 | 0.0019575 | 3/3 |
| 4 | 0.1 | 512 | 0.045805 | 0.00030623 | 3/3 |

ALS improves population error in46/54pairedcases, but is not universally better. For genericfour-factor noiseless targets atN512, median error drops from.0467 to numericalprecision and recovery is3/3. AtnoiseSD.1,N512, ALSmedian.000306 compares withOLS.000624. On noisyone-factor targets atN512, the overparameterized ALSmodel median.000104 is below the prior correctly shared SGDmodel.000421; the earlier modest sharing advantage was therefore optimizer-dependent. Three seeds do not establish universal superiority, and noisy low-N ALS can overfit badly.

## Numerical evidence of non-global local minima

Two noiselessone-factor teachers atN128 plateau despiteexactblocksolves. Seeds180002/180003 end with trainingMSE.018102/.183994 and populationMSE.323013/.876916. Finalten-sweep training changes are at rounding scale. These failures remain in every aggregate.

For the curvature diagnostic only, normalize factor rows and compensate the globalscale to preserve the function. Gradients of half-trainingMSE have norms4.0e-16/6.7e-16. Three explicit gauge tangent vectors scale one factor up and the fourth down; the Hessian annihilates these to rounding accuracy. After projecting out their span, the nine transverse eigenvalues are positive, with minima.01165/.03281. Central finite differences along the smallest transverse-curvature direction agree with positive curvature. Explicit repeated-teacher-factor solutions in the same parameter family have training loss below1e-25.

The applicable theorem is conditional: if an exact stationary point lies on a smooth free scaling-gauge orbit and its Hessian is positive definite on a transverse slice, it is a strict local minimum modulo that orbit. Choose a local slice to the group action, apply the positive-definite second-derivative test on that slice, and use gauge invariance to cover its neighborhood. It is non-global if a lower-loss feasible point exists.

Our floating-point candidates strongly support those hypotheses but do not certify exact stationarity or an interval-positive Hessian neighborhood. Accordingly these are numerical non-global-local-minimum candidates, not a completed rigorous existence proof. Do not identify nearzero Hessian eigenvalues with an RLCT calculation.

## Decision

The SGD residual errors were substantially optimizer-dependent, but exact coordinate solves do not remove all optimization traps. This bounded control is complete; do not launch an indefinite rescue search. For this product family, reporting sharing/sample efficiency without strong optimizer controls would be misleading. Next return to broader full-data image seed replication and the larger-data benchmark pipeline, while keeping a rigorous certificate of the observed local minima as an explicit theory follow-up. The general reusable-circuit recovery and scalable-complexity questions remain open.
