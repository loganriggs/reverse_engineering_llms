# Shared tensor networks: geometry, learnability, and circuit recovery

This separate CPU-first project implements the user's proposal. The earlier native-MLP research remains paused. Synthetic and initial image studies ran on CPU; the larger SVHN data comparison completed on RTX 5080. The native experiments remain unchanged.

Theory directions: [reuse-aware complexity, rank separation, and gradient-flow proofs](SHARING_COMPLEXITY_AND_FLOW.md).

Start here: [research synthesis](RESEARCH_SYNTHESIS.md), [current evidence and completion map](RESEARCH_EVIDENCE_MAP.md), and [continuation checkpoint](CONTINUATION.md). The bounded optimizer and frozen-transfer panels are complete; the broader scaling and quantitative-recovery objective remains active. Latest scaling: [d16/h8 task coverage versus examples](TASK_COVERAGE_H8_RESULTS.md),60audited GPU fits. More tasks help in4/5paired teachers at fixed inputs, but the equal-label-budget comparison is mixed. No experiment is currently running.

## Thesis and distinctions

Represent a multilinear occurrence contraction as `F(z; u)` and impose fixed linear identifications `z=Sθ`, `u=Rx`. Study parameter tying, repeated inputs, and execution DAG reuse separately. The scientific question is whether sharing helps learn compositional targets beyond parameter count and metric effects, and when incorrect sharing hurts.

There is an essential identifiability limit: computing an identical deterministic state twice with tied parameters is mathematically equivalent to computing it once and reusing it. Outputs, derivatives, and any training procedure using only those derivatives cannot distinguish the two. Their execution costs differ. Function observations cannot uniquely recover the actual execution DAG without additional assumptions or observations. See [the mathematical note](MATHEMATICS.md).

Latest: [hidden information and adaptation risk](HIDDEN_EXTRACTION_RESULTS.md).

Previous: [releasing sharing after training](RELEASED_READOUT_RESULTS.md).

Previous: [shared versus independent dictionaries](SHARED_SEPARATE_READOUT_RESULTS.md); [20 competing hypotheses](RECOVERY_HYPOTHESIS_LEDGER.md).

Previous: [four-feature fixed-budget scaling](FINITE_READOUT_H4_RESULTS.md).

Previous: [fresh-teacher fixed-budget confirmation](FINITE_READOUT_CONFIRMATION_RESULTS.md).

Previous: [finite noisy labels and fixed-budget recovery](FINITE_READOUT_RECOVERY_RESULTS.md).

Previous: [local feature stability and profiled information](READOUT_STABILITY_RESULTS.md).

Previous: [three-feature recovery](READOUT_RECOVERY_H3_RESULTS.md); [generic local identification certificate](READOUT_LOCAL_IDENTIFICATION.md).

Previous: [readout information and component recovery](READOUT_RECOVERY_RESULTS.md); [two-feature scalar identification](TWO_FEATURE_SCALAR_IDENTIFICATION.md).

Previous: [function-preserving factor balancing](BILINEAR_GAUGE_BALANCE_RESULTS.md).

Previous: [alternating block Gauss–Newton](BILINEAR_BLOCK_GAUSS_NEWTON_RESULTS.md).

Previous: [five-seed SVHN data and parameter-budget confirmation](SVHN_CONFIRMATION_RESULTS.md).

Previous: [function fidelity versus hidden-computation recovery](BILINEAR_HIDDEN_RECOVERY_RESULTS.md).

Previous: [exact tangent-rank certificate and local learning coefficient](BILINEAR_TANGENT_CERTIFICATE.md).

Previous: [damped Gauss–Newton and tangent-space diagnostics](BILINEAR_GAUSS_NEWTON_RESULTS.md).

Previous: [layerwise alternation on bilinear networks](BILINEAR_ALTERNATION_RESULTS.md).

Previous: [alternating gradients and exact block flow](ALTERNATING_DYNAMICS_RESULTS.md); [long standard-data control](SVHN_LONG_STANDARD_RESULTS.md); [depth-4 GPU results](SVHN_GPU_RESULTS_DEPTH4.md).

Previous: [depth-3 GPU comparison](SVHN_GPU_RESULTS_DEPTH3.md); normalization geometry and cancellation proofs in [the theory note](SHARING_COMPLEXITY_AND_FLOW.md).

Previous: [SVHN GPU standard versus extra-data comparison](SVHN_GPU_RESULTS.md).

Previous: [SVHN nonuniform-width budget control](NONUNIFORM_BUDGET_RESULTS.md); [larger-data scaling plan](SVHN_EXPERIMENT_PLAN.md).

Previous: [standard SVHN results and independent audit](SVHN_STANDARD_RESULTS.md).

Previous: [fresh-seed full-pool image replication](FASHION_REPLICATION_RESULTS.md).

Previous: [ALS optimizer controls and local-minimum candidates](REUSE_ALS_RESULTS.md).

Previous: [finite-data reuse, noise, and conditioning bounds](REUSE_SAMPLE_NOISE_RESULTS.md).

Previous: [factor recovery, canonical interventions, and conditioning](FIXED_DEGREE_RECOVERY_RESULTS.md).

Previous: [fixed-degree reuse and symmetry-trap experiments](FIXED_DEGREE_REUSE_RESULTS.md).

Previous: [initial-loss-descent learning-rate controls](FASHION_FLOW_RATE_RESULTS.md) and [exact reuse complexity versus flatness](REUSE_COMPLEXITY_AND_FLATNESS.md).

Previous: [image-task gradient-flow and multiplicity controls](FASHION_FLOW_RESULTS.md).

Previous: [full-pool Fashion-MNIST curves and independent audit](FASHION_FULL_RESULTS.md).

Previous: [Fashion-MNIST pilot and architecture restrictions](FASHION_PILOT_RESULTS.md).

Previous: [representative scaling and exact task-span diagnosis](SCALING_RESULTS.md).

Previous: [depth5 prefix alignment and scaling boundary](DEPTH5_PREFIX_RESULTS.md).

Previous: [depth5 basin controls and alignment theory](DEPTH5_BASIN_RESULTS.md).

Previous: [coverage training and deep collapse](COVERAGE_AND_COLLAPSE_RESULTS.md).

Previous: [directional sampling and calibration results](DIRECTIONAL_SAMPLING_RESULTS.md).

Previous: [homogeneous evaluation and depth2–5 feasibility](HOMOGENEOUS_EVALUATION_AND_DEPTH.md).

Previous: [fresh five-teacher confirmation](CONFIRMATION_RESULTS.md).

Previous: [multi-task transfer and graph results](MULTITASK_AND_GRAPH_RESULTS.md), [component-identifiability theorem](GENERIC_BILINEAR_COMPONENT_IDENTIFIABILITY.md), and [generic fibers/local RLCT](GENERIC_FIBERS_AND_LOCAL_RLCT.md).

Previous: [readout and longer-training results](READOUT_AND_OPTIMIZATION_RESULTS.md) and [singular readout geometry](SINGULAR_READOUT_GEOMETRY.md).

Earlier findings: [regression controls and d=8 results](REGRESSION_AND_D8_RESULTS.md), [polynomial quotient geometry](POLYNOMIAL_QUOTIENT_GEOMETRY.md), and [initial CPU results](RESULTS.md). E0, audited d=4/d=8 E1 studies, and a bounded E2 metric control are complete. Circuit recovery and deeper scaling remain open; see [continuation](CONTINUATION.md).

## Implemented starting point

`experiment.py` implements a homogeneous degree-four scalar network:

```
hL = (A x) * (B x)
hR = hL                        # tied/reused
hR = (AR x) * (BR x)            # independent occurrences
y  = w @ ((C hL) * (D hR))
```

E0 checks the single-layer derivative, linear gradient/Hessian pullbacks with unequal multiplicities, projected metric, tied-DAG versus tied-formula output/gradient equality, scalar degeneracies, polynomial expansion, homogeneous degree, and the early-matrix degree-two dependence. Nonzero reference tensors use relative error; exact-zero references use absolute error. Threshold is 1e-8 in float64.

E1 uses both tied and untied teachers, paired random seeds and nested training inputs, and three students: tied, width-matched untied, and nearest-parameter-budget untied with fixed top width. All use full-batch Adam, the same learning rate, 600 steps, float64, one CPU thread, gradient clipping at 10, and no early stopping or best-seed selection. Report clipping and failures, not just successful losses. This is simultaneous gradient optimization, not alternating least squares or alternating gradient flow.

At default d=4, hidden width 6, top width 4, parameter counts are 100 (tied), 148 (untied width 6), and 100 (untied width 4). Budget matching is exact here but changes hidden width and function class. An untied teacher is not automatically outside the tied student's representable class.

Teacher output is divided by its exact Gaussian RMS calculated from monomial coefficients. There is no intercept or mean subtraction, preserving homogeneity. Initial matrices have Gaussian entries scaled by inverse square-root fan-in. Width-matched students share their initial A/B/C/D/w, with independently drawn right features in the untied model. Output RMS and probe Jacobian norms are recorded; equal weight scales do not imply equal function distributions. `--init matched` copies A/B into the right branch of the width-matched student, providing equal initial functions but different parameter metrics. The narrow budget student is not function-matched by that flag.

Metrics: training/validation/fresh Gaussian test MSE, exact Gaussian population MSE via polynomial moments, relative monomial coefficient error, diagonal-covariance-shift MSE, sparse-input MSE, parameter count, gradients, clipping, and time. Sparse inputs are Bernoulli(0.25)-masked Gaussian scaled by 2, preserving identity covariance. The shifted covariance uses coordinate standard deviations from 0.5 to 1.5. OOD scores are raw MSE under the original target normalization, not normalized separately for each distribution.

All teacher/data/student seeds are recorded by the source and run arguments. Teacher seeds vary across seed indices, so paired runs measure combined teacher/initialization variability, not initialization variability conditional on a single teacher. Training and evaluation inputs are shared between architectures and teacher types within each seed. Test data are never used for optimizer updates or checkpoint selection.

## Run

From this directory, with PyTorch and NumPy installed:

```bash
python experiment.py --mode check --out results/my_e0
python experiment.py --mode train --device cpu --steps 600 --ntrain 256 --seeds 3 --out results/my_pilot
python experiment.py --mode train --device cpu --steps 600 --ntrain 8,16,32,64,128,256,512,1024 --seeds 5 --out results/my_sweep
```

The default teacher argument is `tied,untied`; use `--teacher untied` for that teacher only. Output directories must not exist, preventing accidental overwrite. Each run records source hash, environment, arguments, checks, learning curves, and CSV results. `completed.json` records completion of the requested loop; inspect per-row status for numerical failures.

The degree-four scalar function space has only binomial(d+3,4) monomials: 35 at d=4. Thus even 64 noiseless generic inputs exceed the coefficient count. The added N=8,16,32 conditions are needed to study genuine undersampling at this dimension; larger-N differences may principally reflect conditioning/optimization. This is a small controlled baseline, not evidence of scalable circuit recovery.

## Stages and decisions

1. E0: establish algebra/geometry identities and implementation correctness.
2. E1: bounded CPU pilot then a five-seed, predeclared sample-size sweep. Report paired differences and all seeds. An initial advantage is exploratory, not confirmation. More seeds or independent teacher families are needed for strong statistical conclusions.
3. E2: same representable target and function-matched initialization; compare ordinary tied SGD, multiplicity-corrected SGD, and projected occurrence SGD. The latter two must agree numerically. Then assess free untied dynamics, NTK/Jacobian spectra and tiny Hessians without equating Hessian rank to RLCT.
4. E3: recovery only after defining the observable equivalence class and an identifiable teacher family. Report parameter count, execution nodes/edges, and contraction cost separately. Holdout predictions alone do not recover the true graph.
5. E4: no GPU scaling until the CPU results and controls justify it. The proposed larger sweeps (input dimensions 16–256, widths 16–512, depths 2–5) are future work. The exact coefficient expansion here is deliberately only suitable for small dimensions.

Do not infer universal tying benefits, native-MLP mechanisms, or semantic manifolds from this study.

Analysis and plotting additionally use SciPy and Matplotlib. All saved experiments run on CPU with the environment recorded in their source/protocol artifacts.
