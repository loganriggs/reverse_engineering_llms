# Readout information and component recovery: two-feature control

24 exact-Gaussian-population fits: three independent teachers, two initializations, four supervision arms. Four input dimensions, two bilinear hidden features, linear readout over their three unordered products. This top parameterization is different from the previous factorized C/D/w architecture. All fits use the same SciPy trust-region least-squares settings, analytic coefficient Jacobian, and maximum200 function evaluations. No restarts.

One/two/three arms expose nested independent scalar heads. The intervention arm exposes the first head plus outputs after ablating hidden feature0 or1; the student uses the corresponding labeled masks with the same underlying readout. These are privileged teacher interventions with known node correspondence, not interventions inferred from passive data. Targets are individually Gaussian-RMS normalized, and the objective averages across supervised outputs.

Four further readouts are entirely withheld from representation training. After freezing the learned hidden product features, fit those heads using32 or128 labels, and also compute the exact population-projection floor. No held-out labels choose representation, stopping, hyperparameters or restarts. This noiseless low-dimensional transfer is deliberately simple.

| Supervision | Parameters | Outputs | Function MSE<1e-12 | Both component cosines>1−1e-6 | Max final source MSE |
|---|---:|---:|---:|---:|---:|
| one | 19 | 1 | 4/6 | 4/6 | 0.746725 |
| two | 22 | 2 | 3/6 | 3/6 | 0.00315207 |
| three | 25 | 3 | 5/6 | 5/6 | 0.00249659 |
| interventions | 19 | 3 | 5/6 | 5/6 | 0.334878 |

These are six fits but only three independent teacher draws. Parameter counts and supervision differ, so this is an information intervention, not a parameter-matched architecture comparison. Each near-exact fit also recovers both components and transfers nearly exactly. Failed fits remain visible below rather than being hidden by median transfer error.

| Teacher | Initialization | Arm | Source MSE | Minimum matched component cosine | Mean held-out projection floor | Mean held-out MSE,128 labels |
|---|---|---|---:|---:|---:|---:|
| 215001 | 216001 | two | 5.14858e-06 | 0.558251 | 2.86598e-05 | 3.6742e-05 |
| 215001 | 216002 | interventions | 0.334878 | 0.176276 | 0.0107601 | 0.0140998 |
| 215002 | 216001 | one | 0.0271971 | 0.372829 | 0.546929 | 0.639443 |
| 215002 | 216001 | two | 2.43334e-05 | 0.994151 | 1.47186e-05 | 1.67477e-05 |
| 215002 | 216002 | one | 0.746725 | 0.469473 | 0.581237 | 0.644762 |
| 215003 | 216001 | two | 0.00315207 | 0.276753 | 0.0717303 | 0.0935882 |
| 215003 | 216002 | three | 0.00249659 | 0.401349 | 0.079585 | 0.0874442 |

## What changed our interpretation

A single output recovers components in4/6fits, three outputs and interventions in5/6each, while two outputs succeed in3/6. These small counts are not evidence of a monotone advantage from extra supervision. The new [two-feature scalar-identification argument](TWO_FEATURE_SCALAR_IDENTIFICATION.md) explains why scalar recovery is possible: a generic binary-quadratic readout factors into two rank-four quadratics, identifying the feature pencil; only the original feature lines in that pencil are bilinear. All three teachers satisfy the independence and nondegeneracy assumptions.

This control therefore does not test removal of a generic scalar-output identification ambiguity. Its remaining failures are approximate/local optimization outcomes, not demonstrated non-identifiability. Extra outputs can change the optimization landscape, but larger residuals are not an information-theoretic floor. The symbolic proof has no approximate stability guarantee.

Verification: independently generated monomial products reproduce all fitted coefficient maps; initial/final analytic Jacobians agree with PyTorch differentiation of direct forward evaluation on31fresh inputs. Gaussian coefficient moments replay losses; separate81-node quadrature replays hidden-subspace and component comparisons. All held-out readouts satisfy normal equations, and exact Gaussian transfer errors replay. Source hashes and all24checkpoints are recorded. SciPy termination status0 (budget exhausted) is retained even when function error is tiny; optimizer termination flags do not define recovery.

Next: extend to three hidden features in six inputs, comparing1/3/6 independent readouts and explicitly labeled ablations. The existing full-product-span identifiability theorem applies to six spanning readouts, while the two-feature scalar factorization argument no longer automatically applies. Preserve exact population objectives and whole held-out heads so optimization, information, and transfer remain separately measurable.
