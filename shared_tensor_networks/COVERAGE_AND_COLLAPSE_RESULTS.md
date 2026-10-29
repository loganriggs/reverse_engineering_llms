# Coverage helps depth3, but depth5 loses predictive signal

Twelve paired CPU fits tested whether informative-direction coverage explains the earlier deep failures. Importance sampling helps atdepth3, but does not rescue depth5 under the tested initialization and optimizer. This is a diagnostic of training, not a result about sharing versus separate computation.

## Design and controls

Three fresh teachers and paired student initializations at depths3/5, d16,h8,6heads. All teachers/students have shared trunks, so every teacher is representable. Each model first receives fixed layer calibration, then per-head output normalization estimated on32768 independent importance-weighted calibration samples. Both training conditions start from identical normalized student states. Training uses1024 fixed examples,600 Adam steps, lr.001, clip10: either uniform sphere draws or a frozen mixture of directions with exact1/q loss weights. There is no division by the sum of weights.

Proposal fitting uses teacher gradients and additional calibration labels. Consequently this is NOT an equal-total-supervision sample-efficiency comparison, even though both training sets have1024 labels. No whole-task holdouts are used in this bounded optimizer diagnostic; task-transfer tests remain a separate obligation. Sphere training keeps the angular objective; homogeneous Gaussian relative population error agrees after exact radial integration, but Gaussian empirical training would weight radii differently.

For each final model, a new64-axis/200-step proposal is fitted on fresh pilot starts. Six independent8192-point batches estimate teacher energy and squared error. Full states, calibration summaries, training inputs/labels/weights and evaluation inputs/weights/losses are saved. No test feedback selects checkpoints or modifies training.

## Results

Relative error1 is zero prediction. Results remain visible when precision gates fail:

|Depth|Teacher seed index|Uniform|Importance|Precision gate|
|---|---:|---:|---:|---|
|3|0|1.26754|0.91354|both pass|
|3|1|1.02135|0.62934|both pass|
|3|2|0.97933|0.64616|both pass|
|5|0|1.02006|1.000009|both fail|
|5|1|1.10539|1.000007|uniform fails|
|5|2|1.07076|1.000009|importance fails|

The descriptive precision gate requires batchCV<10% for both absolute energy and error estimates. Four of six depth5 fits fail; do not interpret their small score differences as precise. The near-unit relative errors have strongly correlated numerator and denominator, but passing a ratio-based diagnostic would not repair the failed absolute-energy check.

Depth3 importance beats uniform in all three pairs; no universal or population-of-teachers significance claim follows from three teachers. At depth5, uniform training target energies are extremely small, while importance training sees far more target signal. Nevertheless importance training loss flattens early near its target energy, rather than learning it. All12 fits complete600steps without nonfinite failures. Early clipping occurs in some runs, especially depth5; no clean optimizer-mechanism attribution yet.

## Is it only output scale?

Post-result diagnostics use three evaluation batches to fit one scalar multiplying all predictions, then test on the other three, and reverse. This is explicitly diagnostic evaluation fitting; it never changes the trained models or primary scores.

Depth5 importance-trained prediction energy is7.5e-6 to9.0e-6 of target energy. Cross-fitted rescaling gives error essentially1 on all three teachers. Uniform-trained depth5 output energy is.020–.105 of target energy, but its scalar-corrected errors are also essentially1. Thus these predictions show little usable linear alignment with the teacher on these evaluated directions; this is more than an amplitude mismatch. For depth3 importance models, cross-fitted scalar errors are.752/.600/.616, consistent with preserved predictive signal. Precision caveats persist.

## A theoretical mechanism to test

Input degree and parameter degree are different. For this architecture, simultaneously multiplying every trainable factor by t, while holding calibration buffers fixed, gives

f(t theta,x)=t^D f(theta,x), D=2^(depth+1)-3.

The trunk recurrence is e_0=0,e_l=2e_(l-1)+2, and the final full-quadratic readout has degree2e_(depth-1)+1. Hence depths2/3/4/5 have parameter degrees5/13/29/61, versus input degrees4/8/16/32. An initial draft incorrectly omitted a quadratic occurrence in this recurrence; the numerical identity check rejected it before the corrected formula was recorded.

Along this path, half squared loss is

L(t)=.5 a t^(2D) - c t^D + .5 b,
L'(t)=D a t^(2D-1)-D c t^(D-1),

where a=E||f(theta,X)||², c=E<f(theta,X),Y>, b=E||Y||² with consistent averaging. For D>2, the origin has zero first and second derivatives along the path even when target error is large. Weak alignment c and small scale can yield very slow gradient recovery. Function-level RMS normalization does not remove this parameter-space flatness. Autograd verifies function scaling and directional loss derivatives for all four depths to4.45e-16 absolute error.

This exhibits a possible optimization obstruction, not proof that observed Adam trajectories follow simultaneous parameter shrinkage. Cached execution versus recomputation remains mathematically equivalent; the result concerns parameterization and correlation.

## Verification and next experiment

Independent NumPy replays validate all saved predictions, targets, weighted objectives and evaluation aggregates across12fits, maximum relative error9.96e-16. The weighted readout gradient matches its explicit formula to1.12e-16. Calibration scales are independently recomputed from saved batch energies. Density weights match the previously independently checked proposal. Audit confirms600steps and no nonfinite failure per fit. The post-hoc scale diagnostic is reproducible from full saved states.

Next distinguish initialization-basin failure from aggressive updates and short training: a bounded depth5 comparison at smaller learning rates, plus a teacher-near initialization positive control, on the exact same frozen datasets. Record initial/final objective, gradients, alignment, parameter norms and clipping; fit fresh final evaluation proposals. Preserve total compute and privileged-supervision limitations. Then resume independent-teacher, width/budget and whole-task transfer scaling. No GPU work has started; the full goal remains incomplete.

Scripts: run_coverage_training.py, audit_coverage_training.py, analyze_coverage_collapse.py, check_deep_scale_path.py. Artifacts: results/coverage_training.
