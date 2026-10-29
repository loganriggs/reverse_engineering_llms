# Fixed-scale exact readout elimination

Twenty completed fits: five fresh d8/h4 teachers × two initializations × joint/reduced training. Ten source tasks, four whole-task holdouts,512source examples and128adaptation examples/task, noise SD .05. Unit A/B parameter rows define predictions; raw radial penalty fixes redundant scales. Readout ridge .001 applies identically to both methods. Same initial functions and fitted readout,200function-evaluation cap, no restarts or tuning. See NORMALIZED_READOUT_PLAN.md for frozen protocol.

## Results

Each row averages both starts within a teacher. Errors are exact Gaussian population MSE; transfer averages four held-out tasks, using OLS on the ten frozen hidden-product features.

| Teacher | Joint source | Reduced source | Joint transfer | Reduced transfer |
|---|---:|---:|---:|---:|
| 232101 | 0.000141611 | 0.000141611 | 0.000373547 | 0.000373547 |
| 232102 | 0.000116002 | 0.0402365 | 0.00028748 | 0.0776057 |
| 232103 | 9.55838e-05 | 9.55838e-05 | 0.0003724 | 0.0003724 |
| 232104 | 0.0179209 | 0.0179209 | 0.0246965 | 0.0246965 |
| 232105 | 0.000167897 | 0.0170398 | 0.00058423 | 0.0422913 |

Joint recovers all four teacher quadratic functions to matched absolute Gaussian cosine>.99 in9/10runs; reduced does so in6/10. Matching removes sign, scale and permutation; this does not recover physical execution reuse. Seven paired runs reach essentially the same source objective (including one shared failure); three reduced runs reach substantially higher objectives. No reduced run materially improves the final objective here. Both reduced starts fail on teacher232102, and its second start fails on232105; failures are retained. All runs stop by relative objective tolerance before the cap, which is not a global-optimality certificate.

The earlier three-case unnormalized/unregularized pilot is a different protocol. This panel changes teachers, dimension, normalization and regularization; it does NOT isolate which change reversed the impression. It establishes that eliminating the readout is not reliably beneficial even with identical objective/initial predictions within each pair. No learning-rate tuning or best-start reporting.

## Verification

Six joint/reduced derivative fixtures pass independent autograd (max relative error4.17e-15) and finite differences (max2.71e-10). All20checkpoint polynomial forward maps, source errors, penalized objectives and transfer OLS predictions independently replay. Reduced readouts agree with independent augmented QR solves; maximum normalized normal-equation residual2.49e-14. Raw row norms finish within8.38e-11of1: the earlier scale explosion is absent. Gaussian feature matching uses quadratic coefficient inner products.

## Why the scale control is legitimate

For each nonzero raw row v=r u with r>0 and ||u||=1, predictions depend only on u. The objective is exactly F(u,W)+gamma/2 sum_i(r_i-1)^2. Radial stationarity therefore implies r_i=1; every angular parameter choice has a unique minimizing radius. This penalty fixes redundant representation scales without excluding directions or changing the angular optimum. The normalization derivative is (I-u uᵀ)/r. Zero rows are excluded and explicitly rejected. This is parameter normalization, not RMSNorm on activations.

## A useful local-minimum equivalence

At fixed hidden parameters theta, write Z=[Phi;sqrt(lambda)I], Yaug=[Y;0], U*(theta)=argmin_U ||ZU-Yaug||². Positive lambda makes Z full column rank and U* smooth. Orthogonality Zᵀ(ZU*-Yaug)=0 gives the exact identity

L(theta,U) = L_reduced(theta) + 1/2 ||Z(theta)(U-U*(theta))||_F².

The common radial penalty is included in both losses. Thus every local minimum of the reduced objective lifts to a joint local minimum, and every joint local minimum has U=U* and projects to a reduced local minimum. The converse uses continuity of U*: any nearby improving theta gives a nearby improving joint point. Stationary points likewise correspond by the envelope gradient. With sufficient differentiability, the reduced Hessian is the Schur complement H_theta_theta-H_theta_U H_UU^{-1}H_U_theta.

Consequently elimination does not introduce new objective local minima under these assumptions. It changes parameter space, trust-region geometry and the optimization trajectory. Higher terminal losses here cannot be explained as minima existing only in the reduced formulation. We have not certified that the failed endpoints are exact local minima: stopping tolerance alone does not prove that. Successful joint checkpoints provide explicit lower-loss feasible reduced candidates after readout profiling.

## Decision

Do not scale up reduced-only training on the assumption it is better. Retain both methods. Next informative control is a crossed comparison on the SAME archived teacher/data/starts, separating factor normalization from readout regularization, followed by local curvature diagnostics only if failures persist. This should be bounded; broad optimizer tuning is not a substitute for feature recovery and held-out transfer. The broader goal remains active.
