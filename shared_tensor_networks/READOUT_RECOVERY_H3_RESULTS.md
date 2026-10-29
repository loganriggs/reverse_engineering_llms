# Three-feature readout recovery

24 exact-population fits: three fresh teachers218001–3 and two initializations219001–2, three bilinear hidden features in six inputs. One/three/six nested independent output heads, or one head plus three individually labeled hidden-node ablations. SciPyTRF with analytic coefficient Jacobian, maximum200function evaluations, fixed1e-12tolerances, no restarts. The six-head teachers span all six hidden products and have six independent input factors; the auditor verifies these theorem assumptions.

Each target is Gaussian-RMS normalized and loss averages over supervised outputs. Four other heads remain excluded from representation training; afterward fit their readouts on frozen hidden products using32/128labels, plus an exact population projection floor. Intervention masks have known node correspondence and are privileged supervision. Top readouts are linear over hidden products, unlike the earlier factorized C/D/w architecture.

| Supervision | Parameters | Outputs | Function MSE<1e-12 | All component cosines>1−1e-6 | Max final source MSE |
|---|---:|---:|---:|---:|---:|
| one | 42 | 1 | 3/6 | 3/6 | 0.0221852 |
| three | 54 | 3 | 6/6 | 6/6 | 5.81227e-25 |
| six | 72 | 6 | 5/6 | 5/6 | 0.013784 |
| interventions | 42 | 4 | 6/6 | 6/6 | 2.24927e-25 |

These are six fits per arm but only three independent teachers. More supervision changes parameter count, residual dimension and optimizer work; this is not an equal-compute or parameter-matched comparison. Every near-exact fit recovers all three components and transfers nearly exactly. The remaining outcomes are retained below.

| Teacher | Initialization | Arm | Source MSE | Minimum matched component cosine | Mean held-out projection floor | Mean held-out MSE,128 labels |
|---|---|---|---:|---:|---:|---:|
| 218001 | 219001 | six | 0.013784 | 0.420368 | 0.0195721 | 0.0212448 |
| 218001 | 219002 | one | 0.00138017 | 0.443647 | 0.0410186 | 0.0451969 |
| 218002 | 219001 | one | 0.0221852 | 0.786214 | 0.0602206 | 0.0643439 |
| 218003 | 219002 | one | 0.00266897 | 0.255472 | 0.481537 | 0.53002 |

## Interpretation

Three heads and labeled ablations recover6/6, versus3/6for one head and5/6for six heads. This supports a useful supervision effect on these tasks but not a monotone benefit of more heads or a broad statistical claim. A six-head fit fails despite the exact-fit identifiability theorem: information sufficiency does not guarantee optimizer success.

The [local-rank certificate](READOUT_LOCAL_IDENTIFICATION.md) adds a crucial qualification: even one scalar head is generically locally identifiable modulo six scaling freedoms in this d6/h3family. Thus these scalar failures cannot establish a universal continuous identification ambiguity. The certificate does not rule out distant discrete alternatives or certify the precise random teachers.

Independent audits reproduce teacher/learned coefficient maps, initial/final Jacobians using direct-forward PyTorch differentiation, all population losses, component matching with separate729-node quadrature, and held-out regression normal equations/errors. Maximum relative Jacobian discrepancy8.58e-16; loss discrepancy2.78e-16. All24checkpoints, failed/budget-limited outcomes, seeds, source hashes and optimizer diagnostics are retained.

Next: distinguish stability from mere exact identification. Evaluate coefficient-Jacobian spectra after removing scale directions, and test small controlled label perturbations while holding the teacher and optimization basin fixed. This can test whether additional heads constrain weak directions more strongly, without mistaking convergence failures for non-identifiability. Further random restarts to rescue a favored arm are not warranted.
