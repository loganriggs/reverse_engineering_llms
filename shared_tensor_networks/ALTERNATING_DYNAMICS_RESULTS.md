# Alternating updates and exact block gradient flow

54fits:36gradient-step fits and18exact frozen-block-flow fits. Three archived generic quartic teachers, exactly symmetric four-factor initializations, rates/dwell times .03/.01/.003, final per-block time30. All objectives are exact Gaussian population MSE via polynomial-exact quadrature. No new sample noise, no restarts, no validation tuning.

Each sweep updates every factor once; the simultaneous gradient step evaluates predictions once, alternating gradients four times, and exact block flow solves four3x3 affine ODEs. These are matched per-block clocks, not matched wall time or FLOPs. For a switched-flow physical clock, a four-block cycle lasts4eta; rescale time by4 when comparing with simultaneous flow.

| Method | Step/dwell | Median MSE | Median factor spread | Distance to simultaneous flow | Recovery |
|---|---:|---:|---:|---:|---:|
| gradient simultaneous | 0.03 | 0.40550128 | 0 | 1.54715e-05 | 0/3 |
| gradient forward | 0.03 | 0.40428425 | 0.122746 | 0.122752 | 0/3 |
| gradient reverse | 0.03 | 0.40428425 | 0.122746 | 0.122752 | 0/3 |
| gradient random | 0.03 | 0.40534055 | 0.0463813 | 0.0463815 | 0/3 |
| gradient simultaneous | 0.01 | 0.40550129 | 0 | 5.46827e-06 | 0/3 |
| gradient forward | 0.01 | 0.40536881 | 0.0402379 | 0.0402381 | 0/3 |
| gradient reverse | 0.01 | 0.40536881 | 0.0402379 | 0.0402381 | 0/3 |
| gradient random | 0.01 | 0.40549849 | 0.00613295 | 0.00613295 | 0/3 |
| gradient simultaneous | 0.003 | 0.40550130 | 0 | 1.66983e-06 | 0/3 |
| gradient forward | 0.003 | 0.40548948 | 0.0120115 | 0.0120115 | 0/3 |
| gradient reverse | 0.003 | 0.40548948 | 0.0120115 | 0.0120115 | 0/3 |
| gradient random | 0.003 | 0.40550129 | 0.00045397 | 0.000453971 | 0/3 |
| exact block flow forward | 0.03 | 0.40433472 | 0.120285 | 0.120291 | 0/3 |
| exact block flow reverse | 0.03 | 0.40433472 | 0.120285 | 0.120291 | 0/3 |
| exact block flow forward | 0.01 | 0.40537066 | 0.0399723 | 0.0399725 | 0/3 |
| exact block flow reverse | 0.01 | 0.40537066 | 0.0399723 | 0.0399725 | 0/3 |
| exact block flow forward | 0.003 | 0.40548953 | 0.0119877 | 0.0119877 | 0/3 |
| exact block flow reverse | 0.003 | 0.40548953 | 0.0119877 | 0.0119877 | 0/3 |

Cyclic updates break exact factor equality, but the benefit is small at this budget and no fit passes the existing strict factor-recovery criterion. Forward/reverse sweeps agree in prediction error here because identical initialization and a permutation-symmetric product objective make them relabelings; this is not an order-independence theorem for arbitrary architectures. Random order gives weaker persistent splitting in this panel.

Exact block flow shows the same qualitative effect: symmetry breaking is not solely an Euler discretization artifact. Switching whole blocks for finite dwell time itself changes the trajectory. The deviations shrink as dwell time shrinks, consistent with the simultaneous-flow limit after clock rescaling. Simultaneous flow was integrated independently with DOP853 at two tolerances; endpoint discrepancy is below1e-8.

Validation: all54final errors independently replay through polynomial coefficients and analytic Gaussian moments. Initial/final analytic gradients agree with autograd. Five-sweep discrete trajectories agree with independent autograd updates; exact-flow trajectories agree with augmented-matrix exponentials built from autograd block Hessians. Maximum discrepancies are in the saved audit records. No nonfinite fits occurred. These tests support correctness, not general convergence.

Conclusion: finite block switching can release an exact symmetry constraint, but does not by itself deliver useful factor recovery in these runs. This bounded test does not justify claiming alternating dynamics are superior. Next distinguish practical layerwise conditioning effects from this small symmetry mechanism, using independent initializations and the actual bilinear architecture; retain step-size and work controls.
