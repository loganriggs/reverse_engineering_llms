

## H264 — All-head response-only fitting reveals both class error and generalization loss

H262/H263 made progress: reversed head splits and broader training directions did
not rescue shared response transfer. H264 removes two possible obstacles at once
for a capacity diagnostic: supervise ALL9 heads and remove the natural-state loss.
This is deliberately not an unseen-head generalization test. Context/document
transfer remains measured. Fixed H259 affine1152 and quadratic1152 feature maps;
no new basis selection, semantics or predictor head-ID input.

Train on full-edit actual-source missing-response targets from768 documents,
4 sites and9 heads =27,648 response pairs. Validate on256 distinct discovery docs;
test on the same128 H259 docs used in recent diagnostics. Test documents never fit
coefficients, but this is an adaptively reused evaluation panel. Ridge grid
0/1e-6/1e-4/.01/.1/1, selected solely by validation response MSE. No refit. Response-
only intercept is unidentifiable from differences, so set it afterward using natural
TRAIN residual mean; it cannot affect any response prediction. Full and10%-edit
actual/angle tests use the same common-native-baseline mediation as H260.

**Empirical projection audit.** Double-precision Gram eigensolve retains all1152
feature directions for BOTH families at relative cutoff1e-12. Condition numbers
9496 affine,11102 quadratic. Maximum normal-equation relative residual1.55e-14;
float32-deployed training fidelities agree with double results to rounding.
Unregularized fit therefore measures the empirical least-squares projection into
this fixed feature-to-write LINEAR readout class. It is not a bound on arbitrary
nonlinear functions of these features, alternative input bases, larger quadratic
maps, or the model's intrinsic computation dimension.

| Feature family / fit | Train response fidelity | Validation | Test, full actual edits |
|---|---:|---:|---:|
|Affine, unregularized|42.28%|22.42%|22.77%|
|Affine, selected ridge .01|41.85%|23.68%|24.12%|
|Quadratic, unregularized|52.80%|37.42%|36.12%|
|Quadratic, selected ridge .1|50.23%|38.86%|37.87%|

The locked 'training fidelity below50%' limited-capacity gate FAILS: quadratic
achieves52.8%. Do not change the threshold to call it a success. Nonetheless47.2%
of training missing-response energy is outside the fitted feature-response column
space on these observations. Improving training fit beyond this numerical projection
requires changing that class/features, not merely choosing a better optimizer.
The >=15-point unregularized train-validation gap gate PASSES: quadratic gap15.38
points; affine gap19.86. This gate is a descriptive finite-panel contrast, not a
bootstrap significance claim. Evidence supports both residual class error and
finite-sample/context generalization loss; it is not an exclusive either/or result.

Selected-response-transfer gate FAILS: required>=50% test fidelity at both scales,
observed37.79% small and37.87% full. Full test interval[34.96,41.08]%. All-head
training is much better than H261's unsupervised-head transfer but not complete.
Natural-state validation residual fidelity for selected quadratic is17.19%; its
unregularized counterpart is−27.41%. Affine values−90.57% and−142.67%. Response
fit is not a guarantee of a good intact replacement, and no native behavioral-
repair claim is made from this capacity experiment.

**Behavior and control directions.** Selected quadratic conditional KL improves
34.19% small/31.83% full relative to the compiler-only response patch. Full actual
logit-effect fidelity92.99%, CE-effect fidelity72.02%. These are conditional mediation
metrics, with a native baseline, not full replacement accuracy. Angle-control
missing-response fidelity is only1.58% small/1.97% full, and conditional KL improves
5.85%/3.43%. Even all-head response fitting follows actual source/context statistics
much better than arbitrary same-span, norm/angle-matched directions. This further
limits claims of universal local geometry or source-span coverage.

All12 ridge fits are retained, alongside validation choices and unregularized
controls. Five relevant tests pass, including the response eigensolve against direct
SVD least-squares on a rank-deficient toy problem, plus joint-fit/patch/feature tests.
Full project suite not rerun. Figure inspected. All H264 runs complete; original
model and frozen feature maps unchanged.

**Next information-gain experiment.** A predeclared2x2 of quadratic input width32/64
and training documents256/768, with624 linear features held fixed, can separate
feature expansion from data effects. Report actual parameter/read counts: widening
quadratic products adds capacity and is NOT a matched-budget win. Select ridge on
a fixed discovery-validation set; report train, validation and frozen test response
fidelity, including angle controls. If added width mainly raises train fit but
not validation, estimation dominates that expansion; if validation rises at fixed
data, the old map omitted useful interactions. Preserve both axes, do not pick the
best cell and call it a fundamental unit. Then use resulting residual diagnostics
to choose tensor-derived partner reads rather than blindly extending PCA products.
Broad research goal remains active.
