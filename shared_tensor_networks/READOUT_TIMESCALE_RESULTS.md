# Readout timescale: finite-time dynamics results

Thirty two-timescale flows, three reduced reference flows, and tighter-tolerance replays completed. Three fresh d3/h2 teachers; both initially optimal and perturbed readouts; five epsilon values. See READOUT_TIMESCALE_PLAN.md for the fixed design and documented stiff-solver amendment. No training/transfer ranking is claimed.

Here epsilon is the inverse readout speed relative to hidden learning: smaller means faster readout adaptation. Distances compare hidden coordinates at time .5 with reduced gradient flow from the same hidden initialization. Each table value is a median over three teachers, not a confidence interval.

| Initial readout | epsilon | Hidden endpoint distance | Final readout tracking error |
|---|---:|---:|---:|
| optimal | 0.1 | 0.3700309 | 0.07356504 |
| optimal | 0.03 | 0.210599 | 0.01857544 |
| optimal | 0.01 | 0.0857169 | 0.006180055 |
| optimal | 0.003 | 0.02779724 | 0.001751387 |
| optimal | 0.001 | 0.009344174 | 0.0005690275 |
| perturbed | 0.1 | 0.6299126 | 0.5927668 |
| perturbed | 0.03 | 0.1267868 | 0.01693837 |
| perturbed | 0.01 | 0.08065728 | 0.005497162 |
| perturbed | 0.003 | 0.03380876 | 0.001665595 |
| perturbed | 0.001 | 0.01169103 | 0.00055875 |

Endpoint distance decreases at every tested speed increase in 6/6 teacher/initial-readout sequences. Changing epsilon from .1 to .001 reduces this distance by factors 13.1–111. This finite panel supports the conditional tracking limit; it does not establish an asymptotic rate or uniform theorem constants.

All 3,030 stored observations pass independent literal-forward loss replay and SVD optimal-readout tracking replay. Sixty endpoint checks compare gradients against Torch autograd and optimal-readout derivatives against centered differences. Maximum gradient error 3.15e-13; maximum derivative error 2.25e-09. Maximum full-path tight-tolerance discrepancy 4.28e-10.

The 3 completed explicit-solver paths agree with Radau to maximum absolute discrepancy 5.55e-08. Explicit outputs and the interrupted attempt log are retained. Solver agreement is not an interval-certified solution.

## Interpretation and next test

The mathematical result identifies a conditioning-sensitive tracking scale, epsilon times the motion of the optimal readout divided by a readout curvature lower bound. Even an initially optimal readout immediately lags when the hidden representation moves. Faster adaptation approaches reduced gradient flow under the stated local uniform assumptions; this does not imply it approaches a better representation. The previous recovery experiments already show that exact elimination can underperform joint training.

The next dynamics comparison should hold the teacher/data/initial function fixed and test finite readout speeds against joint and reduced baselines using feature recovery and whole-task transfer. Finite-dwell alternating flow must remain a separate arm. Use measured gradient work and wall time, not adaptive ODE solver evaluation counts as a substitute for a practical training benchmark. Larger-width task/input scaling remains open; this finite-time check does not satisfy that requirement.
