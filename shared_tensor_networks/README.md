# Shared tensor networks: geometry, learnability, and circuit recovery

This separate CPU-first project implements the user's proposal. The earlier native-MLP research remains paused. No model downloads, GPU use, or changes to the native experiments are needed.

## Thesis and distinctions

Represent a multilinear occurrence contraction as `F(z; u)` and impose fixed linear identifications `z=Sθ`, `u=Rx`. Study parameter tying, repeated inputs, and execution DAG reuse separately. The scientific question is whether sharing helps learn compositional targets beyond parameter count and metric effects, and when incorrect sharing hurts.

There is an essential identifiability limit: computing an identical deterministic state twice with tied parameters is mathematically equivalent to computing it once and reusing it. Outputs, derivatives, and any training procedure using only those derivatives cannot distinguish the two. Their execution costs differ. Function observations cannot uniquely recover the actual execution DAG without additional assumptions or observations. See [the mathematical note](MATHEMATICS.md).

Current findings: [initial CPU results](RESULTS.md). E0, two audited E1 sweeps, and a bounded E2 metric control are complete; E3–E4 are not.

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
