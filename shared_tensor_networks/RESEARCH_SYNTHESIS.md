# Shared tensor networks: what the experiments establish

The research separates parameter tying, repeated inputs and reuse of computed states. These often appear together in a bilinear network, but they impose different mathematical constraints. The strongest outcome is a set of conditional explanations and controls, not a universal claim that sharing improves learning.

## Main findings

**Sharing changes geometry as well as capacity.** A shared parameter receives gradients from every occurrence. Dividing by multiplicity exactly matches projected occurrence-space SGD; ordinary tied SGD uses a different metric. On Fashion-MNIST, calibrating initial descent substantially removed the apparent corrected-SGD slowdown. See [geometry](SHARING_COMPLEXITY_AND_FLOW.md) and [image control](FASHION_FLOW_RATE_RESULTS.md).

**Small reusable circuits need not have small tensor rank or favorable learning dynamics.** Repeated-squaring constructions separate circuit size from formula size and tensor flattening rank. Explicit scalar gradient flows show that a small circuit can still have very flat learning dynamics. These examples vary degree with depth and are not fixed-degree causal evidence against sharing. See [complexity](SHARING_COMPLEXITY_AND_FLOW.md) and [solved family](REUSE_COMPLEXITY_AND_FLATNESS.md).

**Diverse supervision can help recover useful features, but its benefit depends on the label budget.** In the five-teacher d8/h4confirmation, splitting5120scalar labels across ten heads yields median held-out transfer MSE.000852, versus.310for one head. At1280labels, four heads beat ten (.0278versus.1606). Extra task directions can help identification and optimization while leaving too few observations per task at a small budget. These are exact Gaussian evaluations with whole-task holdouts, not universal task-count laws. See [fixed-budget results](FINITE_READOUT_H4_RESULTS.md).

**Sharing can help transfer without every feature being recovered.** In the earlier five-teacher multi-task confirmation, a near-budget wider shared student beats the separate baseline on all five shared-teacher blocks at32/128adaptation labels. Yet component recovery remains partial, and independent-teacher transfer is poor. Width, initialization and optimizer geometry remain confounds in resource-matched comparisons. See [confirmation](CONFIRMATION_RESULTS.md).

**Function matching alone does not recover the computation.** Eight near-exact scalar-output fits failed the teacher hidden-subspace test. An exact polynomial counterexample gives identical outputs with different hidden spans. In a known instrumented graph family, observations alone select the correct physical graph in only20/160cases under a minimal-description preference; the specified internal probes identify160/160. Compression is an explanatory convention, not evidence of physical provenance. See [hidden ambiguity](BILINEAR_HIDDEN_RECOVERY_RESULTS.md) and [graph identification](TASK_SPANS_AND_GRAPH_IDENTIFICATION.md).

**Some reusable features are identifiable under strong, explicit assumptions.** Independent teacher factors, complete task-product span, exact function equality and a fixed minimal bilinear width identify components up to scale/permutation. A new [compactness argument](BOUNDED_APPROXIMATE_IDENTIFICATION.md) gives qualitative approximate stability when normalized factors and readouts lie in a bounded set. It provides no usable numerical error threshold, no overcomplete guarantee and no guarantee gradient training finds the solution.

See [the consolidated learning-dynamics guide](LEARNING_DYNAMICS_SYNTHESIS.md) for current B results, later conditioning/intervention evidence, and the limits of optimizer comparisons.

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

The current bounded experiments are complete and published in the repository with checkpoints, protocols and audits. The broader objective remains active because the proposal's full scaling envelope and quantitative robust-recovery claims are not established. Historical status superseded by the latest update below.

The most useful next learning experiment is to test sufficient task coverage and more source examples at a larger still-tractable fixed width, before jumping to depth5or width512. The old h8/h16scaling study used only12source tasks, below the36/136product dimensions, and a fixed600step training recipe. The recent hidden-transfer diagnosis shows incomplete learned dictionaries; it does not distinguish insufficient source information from optimization and model-selection limits. A fixed-width, task-count/sample-count experiment with exact evaluation is more informative than another optimizer rescue sweep.

Before launching that study, freeze resource budgets, complete-task holdouts, random starts, teacher-normalization privileges and the stopping rule. Quantitative approximate identification remains a separate theoretical target; proving a qualitative compactness limit does not complete it. Novelty relative to the literature has not been assessed, and this report makes no novelty claim.

The complete task-product-span assumption is sufficient, not known necessary in the larger family. Having12tasks below36/136products does not by itself prove the hidden features cannot be identified or learned; the next experiment must test that hypothesis rather than assume it.

Subsequent result: TASK_COVERAGE_H8_RESULTS.md reports60fresh d16/h8fits.36tasks beats12in4/5teachers at fixed input counts, but only2/5at equal total labels. The proposed information-budget question has now been tested at this width, with a conditional rather than universal benefit. Larger-width scaling remains.

Latest scaling update: TASK_COVERAGE_H16_RESULTS.md adds60d32/h16fits. At equal69632labels,34tasks×2048inputs beats136tasks×512inputs in5/5teachers. This supports input coverage alongside task diversity, while all low-input random fits have explicit lower-loss teacher-start candidates. The proposed larger envelope is still not fully executed.

## Readout learning timescale extension

[READOUT_TIMESCALE_RESULTS.md](READOUT_TIMESCALE_RESULTS.md) completes 30 finite-time two-timescale flows and independent checks. Faster readout adaptation approaches reduced gradient flow in all six tested teacher/initialization sequences. [READOUT_TIMESCALE_THEORY.md](READOUT_TIMESCALE_THEORY.md) states the conditioning-sensitive tracking bound and uniform local assumptions. This is an ODE mechanism result, not evidence that faster readout learning improves feature recovery or transfer. Finite-dwell alternating recovery tests and larger-width scaling remain open.

## Finite readout speed and recovery

[READOUT_SPEED_RECOVERY_RESULTS.md](READOUT_SPEED_RECOVERY_RESULTS.md) reports 70 paired fits: moderate-speed exact readout-flow splitting improves aggregate transfer over joint speed1 on all five archived teachers, but beats reduced training on only three. Fast explicit joint steps lose hidden-learning time to stability-driven backtracking. These are finite-budget results with independent metric audits; fresh-teacher and matched-time confirmation remain open.

## Fresh-teacher readout-speed result: advantage did not replicate

[READOUT_SPEED_CONFIRMATION_RESULTS.md](READOUT_SPEED_CONFIRMATION_RESULTS.md) adds 70 fresh-teacher/fresh-initialization fits, all at accumulated hidden-step time20. Split speed10 loses its previous aggregate advantage: transfer median .2517 versus joint1 .0913 and reduced .1906; wins only2/5 and1/5 teacher blocks respectively. These primary arms already had matching clocks in the old panel. High-speed joint and split methods have similar aggregate errors but substantially different gradient work. This supports stiff-block integration as a computational tool, not a reliable feature-recovery improvement. Larger d64/h32 scaling remains open.

## d64/h32 scaling and sampled information

[TASK_COVERAGE_H32_RESULTS.md](TASK_COVERAGE_H32_RESULTS.md) completes 60 audited GPU fits and 240 adaptation fits. At equal labels, more inputs/fewer tasks wins all five teachers (.0483 versus1.5217 median transfer). Every teacher start recovers all32components; random starts recover zero full sets at low input coverage and3/10at high coverage. Thirty-six of40random endpoints have explicit same-condition lower training-loss witnesses. This extends the coverage/optimization distinction to larger width without proving a universal scaling law.

[SAMPLED_HIDDEN_INFORMATION.md](SAMPLED_HIDDEN_INFORMATION.md) derives profiled finite-sample rank bounds, including sampled hidden-value basis changes that readouts absorb. Sixty-three derivative/rank fixtures and three finite basis-change tests pass. The bounds do not explain the current grid by a simple shortage of constraints; conditioning and optimization remain material. d128/h64 requires verified memory/storage adaptation before launch; the broad goal remains active.

## Verified infrastructure for d128/h64

[ACTIVE_COVERAGE_RESULTS.md](ACTIVE_COVERAGE_RESULTS.md) verifies independent-case grouping against the original full-batch updates and tests all four larger shapes on GPU (largest allocated6.62GiB). Deterministic block regeneration and lossless checkpoint chunks pass. An initial post-check shutdown abort is retained; cleanup replay exited successfully. This prepares larger scientific runs; it supplies no new recovery result at h64.


## Exact sampled-information extension; h64 training active

SAMPLED_HIDDEN_INFORMATION.md now includes exact rank certificates for d4/h2 and three product tasks: seven generic inputs leave two local non-scaling ambiguities, whereas eight give local identification modulo scaling. Thirty integer fixtures retain nongeneric small-range deficits alongside broader-range witnesses. This supplies no conditioning, noise or global-recovery guarantee. The d128/h64 producer was directly confirmed live; evaluation/audit and scientific results remain pending. Aggregation code has fixture tests but no h64 scientific outcome is claimed.


## Profiled information and conditioning

PROFILED_INFORMATION_AND_CONDITIONING.md adds fixed-weight information monotonicity, exact linearized noise risk, a local reduced-flow rate and a certified path showing arbitrarily poor conditioning despite full rank. Checks cover60 configurations,85 PSD comparisons and15 linear-noise simulations; no nonlinear recovery experiment is claimed. This motivates measuring spectra rather than relying only on dimension counts. The h64 producer remains live; scientific outcomes remain pending.


## Local nonlinear noise check

LOCAL_NOISE_RECOVERY_RESULTS.md reports450 privileged teacher-start local fits with independent audit. All150 paired noise sequences approach their linearized hidden-coordinate prediction as noise decreases. This supports the conditioning interpretation locally, not recovery from random initialization or global convergence. The h64 producer remains active; full scientific evaluation is pending.


## Depth-three component ambiguity

DEPTH3_INTERNAL_MIXING.md supplies an exact width-two counterexample: complete product-task functions do not identify second-layer feature axes. Legal mixtures preserve all output functions but change coordinate-ablation effects. Four rational examples and a rank11/17parameter certificate show two local mixing freedoms beyond four scales. This refines deeper recovery criteria; it is not a general-width impossibility theorem or an explanation of all historical deep-learning failures. h64 training remains live (27/60 models saved at latest check).


## Which intervention identifies deep axes?

DEPTH3_INTERVENTION_IDENTIFICATION.md proves that one named internal deletion with two informative output measurements resolves the width-two mixing ambiguity. Eight exact Jacobian certificates distinguish first-order from higher-order information. One squared-task readout provides a quartic weak signal along a mixing direction; the mixed task provides a quadratic signal. Constrained scalar-flow identities are verified, not claimed as full-network dynamics. Larger h64 training remains live,30/60models saved at latest check.


## Training with deep causal supervision

DEPTH3_CAUSAL_TRAINING_RESULTS.md reports20 audited fixed-firstlayer population fits. Adding ablation observations improves axis recovery1/10→3/10 and lowers ablation error on5/5teacher fixtures, but worsens ordinary error on5/5 and reduces exact ordinary fits5/10→3/10. All near-exact observation-only fits recover the feature plane, while axes remain ambiguous. One capped case retained. This distinguishes useful additional information from reliable optimization; no general deep-recovery claim follows. h64 remains running.


## Audited width-64 extension: task diversity depends on input coverage

The completed [h64 report](TASK_COVERAGE_H64_RESULTS.md) extends the same depth-two shared-feature family to d128/h64. All60 training fits and scientific evaluations passed the independent audit. At equal scalar-label budget, allocating labels to more input examples rather than more task heads wins4/5 teachers, following5/5 at h16 and5/5 at h32. These are separate five-teacher panels, not a controlled width-only comparison.

| Hidden width | High input count | Low / full task count | Transfer MSE, low / full tasks | Complete feature recovery, low / full tasks |
|---:|---:|---:|---:|---:|
|16|2048|34 /136|.0322085 / .0134433|2/10 /6/10|
|32|8192|132 /528|.048311 / .0267005|3/10 /3/10|
|64|32768|520 /2080|.0160153 / .0237113|4/10 /3/10|

Transfer errors are medians of teacher means, averaging two random starts and four whole-task holdouts. Primary adaptation sample counts scale with width and differ across panels. Recovery demands every matched feature exceed cosine.99, so its criterion becomes stricter as width grows. Do not fit a sample-complexity exponent to this table.

At h64, adding tasks improves paired transfer on5/5 teachers at8192inputs and3/5 at32768inputs. The latter majority coexists with a worse aggregate median: teacher heterogeneity matters. All20 privileged teacher-start controls recover all64features;38/40 random fits have same-condition lower empirical-loss witnesses. This establishes remaining optimization difficulty but not strict local minima, nor that all generalization error is optimization error.

Next scaling decision: preserve adequate input coverage and whole-task holdouts; do not simply multiply task heads. Before d256/h128 training, perform a memory/work preflight covering both data generation and independent population/transfer analysis. Its8256 product coordinates make dense readout and SVD costs substantially larger. Choose the feasible training layout from that measurement and verify any streamed gradients against the existing full-batch implementation before freezing a scientific protocol. No additional optimizer rescue is justified by these data alone. Depth3–5 learned recovery remains a separate incomplete requirement.
