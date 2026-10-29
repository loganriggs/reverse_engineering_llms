# Task coverage versus example count at d16/h8

60completed GPU fits on five fresh shared teachers. Source tasks12/36crossed with inputs512/1536; two random starts and one privileged teacher start each. All train3000full-batch Adam steps atlr.003with per-model clip10, float32/TF32off. No restarts or held-out selection. Exact Gaussian float64 evaluation and four whole held-out tasks with32/128noisy adaptation labels. See TASK_COVERAGE_H8_PLAN.md.

## Primary128label transfer

MSE targets have unit Gaussian second moment. Random medians aggregate both starts within each teacher first; component threshold requires all eight matched absolute cosines>.99. Teacher starts are diagnostics, not a practical learning method.

| Inputs | Tasks | Scalar labels | Initialization | Median teacher transfer MSE | All8components recovered |
|---|---|---:|---|---:|---:|
| 512 | 12 | 6144 | random | 0.0585504 | 6/10 |
| 512 | 12 | 6144 | teacher | 0.00436119 | 5/5 |
| 512 | 36 | 18432 | random | 0.00471626 | 8/10 |
| 512 | 36 | 18432 | teacher | 0.00399584 | 5/5 |
| 1536 | 12 | 18432 | random | 0.0247131 | 5/10 |
| 1536 | 12 | 18432 | teacher | 0.00390241 | 5/5 |
| 1536 | 36 | 55296 | random | 0.00451732 | 8/10 |
| 1536 | 36 | 55296 | teacher | 0.00385681 | 5/5 |

## Every random-start teacher aggregate

| Teacher |512×12|512×36|1536×12|1536×36|
|---|---:|---:|---:|---:|
| 239101 | 0.0654139 | 0.0038614 | 0.00390684 | 0.0755079 |
| 239102 | 0.20868 | 0.0289588 | 0.0247131 | 0.00368305 |
| 239103 | 0.0585504 | 0.0981634 | 0.0821666 | 0.00394859 |
| 239104 | 0.00376817 | 0.00333708 | 0.0717406 | 0.0260549 |
| 239105 | 0.00509162 | 0.00471626 | 0.00454968 | 0.00451732 |

At each fixed input count,36tasks beats12tasks in4/5teacher blocks. This adds labels and readout parameters; it is not a pure diversity intervention. At the equal18432label budget,36tasks×512examples beats12tasks×1536examples in only2/5teachers. The lower median(.00472versus.0247) does not establish a consistent paired advantage. All individual outcomes are retained rather than reporting the favorable median alone.

Increasing input count improves transfer in3/5teachers at either task count. It need not improve finite-budget optimization monotonically. All20teacher-start runs pass the feature-recovery threshold, while random recovery varies5–8/10. The teacher-start gap is evidence of initialization dependence under this recipe, not a proof each poor random fit is a strict local minimum.

Both12and36tasks can recover all eight features in individual random runs. The full36product-span condition is sufficient for the exact theorem, not necessary for every target. Task coverage, input coverage, extra readout parameters, noise averaging and optimizer dynamics are not fully disentangled by this grid.

## Implementation and audit

GPU was idle before launch (RTX5080, no compute applications). Models were batched independently with separate parameter/moment slots. Masked loss is normalized separately by each model's own scalar-label count; gradients are clipped per model. A five-step float64 check matches separate Adam optimizers to1e-12tolerance. CPU/CUDAfloat32 forward and gradient relative errors are1.13e-7and1.72e-7. Inactive readout rows remain exactly unchanged and are excluded from reported active counts (688/1552parameters); equal allocated tensor shapes are not claimed as equal active model sizes.

All60source fits and240learned/oracle adaptation fits independently replay. Literal bilinear forwards verify training losses and target/noise data; SVD verifies adaptation pseudoinverses, including the32label underdetermined case. Gaussian quadratic coefficient inner products independently verify component cosines against the producer's matrix-trace metric. Exact Wick cross-Grams evaluate source/transfer errors; that engine has separate quadrature validation. Dataset/checkpoint hashes, all15logged200-step loss/gradient records per teacher batch, and exact initial pairing across conditions pass. No nonfinite run was dropped.

Teacher and random initial task RMS use exact distributional normalization, an explicit simulation privilege. Teacher factors/readout ranks are checked. Primary128label adaptation has more observations than36features; secondary32label minimum-norm fits depend on the chosen feature coordinates. No generic sample-complexity bound follows.

Runtime is recorded per12-model GPU batch; summing the duplicated per-case batch time would overcount compute. This is not a timing benchmark or a comparison with earlier CPU/TRF work. The previous larger scaling study also differs in input distribution, initialization/calibration, noise, task count and training duration, so this experiment cannot attribute its better recovery to task count alone.

## Decision

Close this frozen grid without tuning the losing cases. It establishes a tractable larger feature-recovery setting and a conditional task-diversity signal, but not a consistent equal-label-budget advantage. Next scaling should preserve the crossed design, active-parameter accounting and whole-task holdouts at a larger width; it should not treat the low median as a confirmed universal benefit. Quantitative robust-identification bounds remain a separate theoretical question. The overall research goal remains active.
