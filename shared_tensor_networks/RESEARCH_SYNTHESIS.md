# Shared tensor networks: what the experiments establish

The research separates parameter tying, repeated inputs and reuse of computed states. These often appear together in a bilinear network, but they impose different mathematical constraints. The strongest outcome is a set of conditional explanations and controls, not a universal claim that sharing improves learning.

## Main findings

**Sharing changes geometry as well as capacity.** A shared parameter receives gradients from every occurrence. Dividing by multiplicity exactly matches projected occurrence-space SGD; ordinary tied SGD uses a different metric. On Fashion-MNIST, calibrating initial descent substantially removed the apparent corrected-SGD slowdown. See [geometry](SHARING_COMPLEXITY_AND_FLOW.md) and [image control](FASHION_FLOW_RATE_RESULTS.md).

**Small reusable circuits need not have small tensor rank or favorable learning dynamics.** Repeated-squaring constructions separate circuit size from formula size and tensor flattening rank. Explicit scalar gradient flows show that a small circuit can still have very flat learning dynamics. These examples vary degree with depth and are not fixed-degree causal evidence against sharing. See [complexity](SHARING_COMPLEXITY_AND_FLOW.md) and [solved family](REUSE_COMPLEXITY_AND_FLATNESS.md).

**Diverse supervision can help recover useful features, but its benefit depends on the label budget.** In the five-teacher d8/h4confirmation, splitting5120scalar labels across ten heads yields median held-out transfer MSE.000852, versus.310for one head. At1280labels, four heads beat ten (.0278versus.1606). Extra task directions can help identification and optimization while leaving too few observations per task at a small budget. These are exact Gaussian evaluations with whole-task holdouts, not universal task-count laws. See [fixed-budget results](FINITE_READOUT_H4_RESULTS.md).

**Sharing can help transfer without every feature being recovered.** In the earlier five-teacher multi-task confirmation, a near-budget wider shared student beats the separate baseline on all five shared-teacher blocks at32/128adaptation labels. Yet component recovery remains partial, and independent-teacher transfer is poor. Width, initialization and optimizer geometry remain confounds in resource-matched comparisons. See [confirmation](CONFIRMATION_RESULTS.md).

**Function matching alone does not recover the computation.** Eight near-exact scalar-output fits failed the teacher hidden-subspace test. An exact polynomial counterexample gives identical outputs with different hidden spans. In a known instrumented graph family, observations alone select the correct physical graph in only20/160cases under a minimal-description preference; the specified internal probes identify160/160. Compression is an explanatory convention, not evidence of physical provenance. See [hidden ambiguity](BILINEAR_HIDDEN_RECOVERY_RESULTS.md) and [graph identification](TASK_SPANS_AND_GRAPH_IDENTIFICATION.md).

**Some reusable features are identifiable under strong, explicit assumptions.** Independent teacher factors, complete task-product span, exact function equality and a fixed minimal bilinear width identify components up to scale/permutation. A new [compactness argument](BOUNDED_APPROXIMATE_IDENTIFICATION.md) gives qualitative approximate stability when normalized factors and readouts lie in a bounded set. It provides no usable numerical error threshold, no overcomplete guarantee and no guarantee gradient training finds the solution.

## What the dynamics experiments add

Alternating gradients, exact block flow, ALS, full/block Gauss–Newton, factor balancing and variable projection were tested separately. None universally wins. Naive layerwise alternation did not justify its extra full-gradient work in the tested panel. ALS and GN exposed substantial optimizer-dependent error, but exact function fit still did not establish circuit recovery.

Readout elimination provides the cleanest recent mechanism test. At an exactly fitted readout, the reduced residual Jacobian is A+B, with orthogonal A and B. Full reduced Gauss–Newton uses AᵀA+BᵀB; eliminating the readout increment from joint Gauss–Newton gives AᵀA. Both have the same gradient. Holding objective, damping and line search fixed, removing BᵀB improves four paired runs, worsens two and matches four. Greater instantaneous descent therefore need not mean better final recovery. All20trajectories/654steps were audited. See [derivation](PROFILED_GAUSS_NEWTON_GEOMETRY.md) and [intervention](PROFILED_CURVATURE_RESULTS.md).

## Scaling exposes two additional bottlenecks

The synthetic scaling study covered d16/32,h8/16,depth2/3, shared and independent teachers, width and approximate parameter-budget controls. Direct hidden-feature evaluation now covers all saved larger models. The12source outputs do discard useful directions, but removing that compression does not broadly rescue transfer.

For the d32depth2shared student, the best population readout error drops only from.961over source outputs to.873over hidden products. Actual128label hidden transfer error is21.85, while true shared-teacher features yield.0694. The first gap concerns learned representation; the second concerns finite-label estimation. These are noiseless adaptation targets, so label-noise amplification alone cannot explain the result. See [exact depth2diagnosis](LARGER_HIDDEN_TRANSFER_RESULTS.md).

At depth3, the stable d16shared-teacher cases also have poor learned hidden transfer, while the true shared dictionary achieves about.0039. Higher-dimensional evaluation is harder: targeted proposal replication improved the weakest regime's original-readout precision count from20/96to80/96, without rescuing transfer. Remaining failures are retained. Feature calibration is another estimator choice: RMS normalization induces a diagonal raw-coordinate ridge penalty. See [depth3](DEPTH3_HIDDEN_TRANSFER_RESULTS.md), [measurement replication](DEPTH3_MEASUREMENT_RESULTS.md), and [penalty identity](CALIBRATION_AS_REGULARIZATION.md).

Depth4/5unprivileged synthetic learning remains difficult. Teacher-prefix controls and evaluation diagnostics establish limits of the tested recipe, not successful scalable recovery. We did not execute the entire proposed d256/h512/100k grid. See [deep controls](DEPTH5_PREFIX_RESULTS.md).

## The image result is useful but answers a different question

On extra-data SVHN at depth3, the five-seed mean accuracy is87.26%for the narrow shared stack,88.96%for the equal-width tree and89.74%for an approximately parameter-matched wider shared stack. The wide stack wins accuracy against the tree in every seed, while the tree wins cross-entropy in every seed. This demonstrates metric and resource-allocation dependence, not circuit recovery or an intrinsic advantage from untying. Official test data were unused. See [SVHN confirmation](SVHN_CONFIRMATION_RESULTS.md).

## What remains open

The current bounded experiments are complete and published in the repository with checkpoints, protocols and audits. The broader objective remains active because the proposal's full scaling envelope and quantitative robust-recovery claims are not established. No current job is running.

The most useful next learning experiment is to test sufficient task coverage and more source examples at a larger still-tractable fixed width, before jumping to depth5or width512. The old h8/h16scaling study used only12source tasks, below the36/136product dimensions, and a fixed600step training recipe. The recent hidden-transfer diagnosis shows incomplete learned dictionaries; it does not distinguish insufficient source information from optimization and model-selection limits. A fixed-width, task-count/sample-count experiment with exact evaluation is more informative than another optimizer rescue sweep.

Before launching that study, freeze resource budgets, complete-task holdouts, random starts, teacher-normalization privileges and the stopping rule. Quantitative approximate identification remains a separate theoretical target; proving a qualitative compactness limit does not complete it. Novelty relative to the literature has not been assessed, and this report makes no novelty claim.

The complete task-product-span assumption is sufficient, not known necessary in the larger family. Having12tasks below36/136products does not by itself prove the hidden features cannot be identified or learned; the next experiment must test that hypothesis rather than assume it.

Subsequent result: TASK_COVERAGE_H8_RESULTS.md reports60fresh d16/h8fits.36tasks beats12in4/5teachers at fixed input counts, but only2/5at equal total labels. The proposed information-budget question has now been tested at this width, with a conditional rather than universal benefit. Larger-width scaling remains.

Latest scaling update: TASK_COVERAGE_H16_RESULTS.md adds60d32/h16fits. At equal69632labels,34tasks×2048inputs beats136tasks×512inputs in5/5teachers. This supports input coverage alongside task diversity, while all low-input random fits have explicit lower-loss teacher-start candidates. The proposed larger envelope is still not fully executed.

## Readout learning timescale extension

[READOUT_TIMESCALE_RESULTS.md](READOUT_TIMESCALE_RESULTS.md) completes 30 finite-time two-timescale flows and independent checks. Faster readout adaptation approaches reduced gradient flow in all six tested teacher/initialization sequences. [READOUT_TIMESCALE_THEORY.md](READOUT_TIMESCALE_THEORY.md) states the conditioning-sensitive tracking bound and uniform local assumptions. This is an ODE mechanism result, not evidence that faster readout learning improves feature recovery or transfer. Finite-dwell alternating recovery tests and larger-width scaling remain open.

## Finite readout speed and recovery

[READOUT_SPEED_RECOVERY_RESULTS.md](READOUT_SPEED_RECOVERY_RESULTS.md) reports 70 paired fits: moderate-speed exact readout-flow splitting improves aggregate transfer over joint speed1 on all five archived teachers, but beats reduced training on only three. Fast explicit joint steps lose hidden-learning time to stability-driven backtracking. These are finite-budget results with independent metric audits; fresh-teacher and matched-time confirmation remain open.
