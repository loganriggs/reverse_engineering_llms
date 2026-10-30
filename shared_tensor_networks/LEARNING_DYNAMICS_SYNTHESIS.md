# B: what sharing changes about learning

The evidence supports three distinct explanations: sharing changes the parameter metric; available observations constrain some hidden directions much more strongly than others; and an identifiable target can still be difficult to optimize. We have not found an alternating update rule that reliably recovers reusable features better across fresh teachers.

This guide consolidates completed panels. Experiment counts below are not independent replications: several panels reuse teachers, starts or trajectory prefixes. The d128/h64 task/input study has now passed its full audit; see TASK_COVERAGE_H64_RESULTS.md for the separate scaling results. The full research objective remains active.

## 1. Geometry: the same function can learn differently

With distinct parameters theta copied into occurrences z=S theta, gradients add across uses. The metric M=SᵀS contains occurrence multiplicities. Ordinary tied gradient flow differs from occurrence-space flow projected onto the tying constraints. Preconditioning by M^-1 makes those two constrained flows agree exactly; the equivalence also holds for compatible block schedules. It does not automatically extend to Adam.

At a common state, freeing the occurrences adds instantaneous descent equal to ||(I-P)grad_z L||², where P projects onto the tying constraints. This is a local comparison, not a global convergence or generalization ordering. In a certified tiny quartic model, tied and untied parameterizations both span the same15-dimensional output tangent space despite having39versus57parameters. Conditioning and paths can matter without a missing local output direction.

The Fashion-MNIST control makes the metric issue concrete: at depth4, corrected SGD reached60.58%at the original learning rate and76.88%after training-only descent calibration; ordinary tied SGD reached75.02%. These are one-epoch, three-seed validation medians, not a settled optimizer ranking.

Sources: [geometry and exact ranks](SHARING_COMPLEXITY_AND_FLOW.md), [image rate control](FASHION_FLOW_RATE_RESULTS.md).

## 2. Alternation: distinguish steps, flow and block solves

Finite alternating steps, finite-dwell switched flow and exact block minimization are different algorithms. Rapid equal-duty switching approaches simultaneous flow after clock rescaling. Persistent advantages must therefore involve finite dwell, different timescales/metrics, or other algorithmic differences.

| Completed comparison | Finding | Limit |
|---|---|---|
| Layerwise alternating gradients | Forward/reverse each beat joint in0/36matched-full-gradient-work pairs | Specialized block implementations could have different costs |
| Exact frozen-block flow | Finite switching broke equal-factor symmetry; no factor recovery in this panel | Small four-factor family; equal block clocks, not equal compute |
| Alternating least squares | Improved population error in46/54product-model cases | Not compute matched; shared early blocks generally are not linear subproblems |
| Full Gauss–Newton | All36endpoints improved over long-gradient baselines | Dense solves cost more; function fit is not circuit recovery |
| Block Gauss–Newton | Some improvements and some regressions | No universal benefit over full GN |
| Factor balancing | Function-preserving coordinates substantially changed some outcomes | Fixed damping after rescaling changes the metric; not a free invariance |

Sources: [layerwise alternation](BILINEAR_ALTERNATION_RESULTS.md), [block flow](ALTERNATING_DYNAMICS_RESULTS.md), [ALS](REUSE_ALS_RESULTS.md), [GN](BILINEAR_GAUSS_NEWTON_RESULTS.md), [block GN](BILINEAR_BLOCK_GAUSS_NEWTON_RESULTS.md), [balancing](BILINEAR_GAUGE_BALANCE_RESULTS.md).

A further exact local comparison is now available: for a positive-definite quadratic in two scalar blocks, finite-dwell exact alternating flow has asymptotic spectral contraction no better than joint flow at equal per-coordinate active time, strictly worse with nonzero coupling. The proof compares traces at equal determinant using convexity of cosh(sqrt(t)). This does not order every finite trajectory or cover vector blocks/nonconvex escape. Matrix-exponential and selected ODE checks pass across480settings. See [two-block rate proof](TWO_BLOCK_FLOW_RATE.md).

## 3. Readout adaptation: easier integration does not imply better features

Faster readout adaptation approaches exactly refitted-readout gradient flow under uniform local conditioning assumptions. Thirty two-timescale flows support that limit. Exact readout elimination nevertheless need not select better hidden representations.

The strongest empirical caution is the failed fresh-teacher confirmation. Moderate-speed split updates initially improved transfer on all five archived teacher blocks, but on fresh fixtures beat ordinary joint training on only2/5. Median transfer MSE was .2517 for split10, .0913 for joint1 and .1906 for exact elimination. High-speed splitting reduced gradient work relative to explicit joint updates; that integration benefit did not establish better recovery.

There is also a precise curvature distinction. Reduced-residual GN uses AᵀA+BᵀB, whereas profiling the readout increment out of joint GN leaves AᵀA. The gradients agree. Removing the extra term improved4paired runs, worsened2and tied4in a controlled20-fit panel. Greater instantaneous predicted descent did not order final outcomes.

Sources: [tracking theorem](READOUT_TIMESCALE_THEORY.md), [flow checks](READOUT_TIMESCALE_RESULTS.md), [fresh confirmation](READOUT_SPEED_CONFIRMATION_RESULTS.md), [curvature identity](PROFILED_GAUSS_NEWTON_GEOMETRY.md), [intervention](PROFILED_CURVATURE_RESULTS.md).

## 4. Information: count, conditioning and uniqueness are different

After refitting free readouts, the observable hidden derivative is A[v]=(I-Pi)D Phi[v]W. At an exact noiseless fit, AᵀA is the profiled Hessian in a specified hidden chart. Its rank measures first-order identifiable directions; its small eigenvalues measure weak information and slow local learning. Neither guarantees discovery from random initialization.

Exact certificates show that the d4/h2three-task model needs eight generic inputs for full first-order identification modulo scales; seven leave two additional local ambiguities. Retained small-integer examples have less information than the generic count. A rational path proves full rank can be arbitrarily ill-conditioned even with bounded inputs and fixed normalized factors.

Appending observations/tasks with their old weights retained cannot decrease the profiled information matrix. This does not compare fixed-label reallocations or mean-loss normalization. Linearized Gaussian noise gives covariance sigma²(AᵀA)^-1. Sixty information fixtures,15linear-noise tests and450privileged teacher-start nonlinear fits support the local interpretation; every150paired nonlinear noise sequence approaches its linear prediction as noise decreases. These are local estimator tests, not global learning results.

Sources: [exact sampled ranks](SAMPLED_HIDDEN_INFORMATION.md), [conditioning and local flow](PROFILED_INFORMATION_AND_CONDITIONING.md), [nonlinear check](LOCAL_NOISE_RECOVERY_RESULTS.md).

## 5. Deeper computation may require internal supervision

An exact width-two depth-three example has two hidden mixing ambiguities beyond scaling, even with all product-task functions known. The alternatives preserve outputs but change coordinate-ablation effects. One specified ablation with two informative task measurements identifies the axes in this family. A less informative measurement gives a quartic loss along a mixing direction, while a mixed-task measurement gives a quadratic loss. The corresponding constrained scalar flows have algebraic versus exponential decay; these are not full-network flow formulas.

Training exposes the remaining gap. In20fixed-first-layer population fits, adding ablation labels improves component recovery1/10→3/10and ablated error on5/5teacher fixtures, but worsens ordinary error on5/5and reduces near-exact ordinary fits5/10→3/10. All five near-exact observation-only fits recover the feature plane, but only one crosses the axis cosine threshold. More informative supervision helps specify the desired computation without guaranteeing the optimizer finds it.

Sources: [ambiguity](DEPTH3_INTERNAL_MIXING.md), [intervention identification](DEPTH3_INTERVENTION_IDENTIFICATION.md), [training test](DEPTH3_CAUSAL_TRAINING_RESULTS.md).

## Current decision

Keep ordinary joint learning as a baseline. Do not reopen closed optimizer sweeps to search for a favorable ranking. Use the completed, audited larger-width task/input study to choose the next scaling step. At equal label budget more inputs won on4/5teachers; additional tasks helped5/5at low input count but3/5at high input count. Feature recovery remains inconsistent from random starts. Deeper studies must separately measure function fit, feature-plane agreement, axis agreement and intervention agreement, with justified identification assumptions. Global convergence, universal sharing advantages, robust deep circuit recovery and the full proposed scaling envelope remain unestablished. No result in this guide is presented as a literature novelty claim.
