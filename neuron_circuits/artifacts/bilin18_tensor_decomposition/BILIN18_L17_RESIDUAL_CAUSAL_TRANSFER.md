

## H260 — Natural-state residual predictors transfer poorly to causal source changes

H259 made progress: quadratic features and sensitivity training both improved
non-oracle natural-state prediction on new documents. H260 freezes all four core
selected models and tests changes in actual L17 attention-head inputs. Earlier
H249 mediation used a common native baseline; this reuses that attribution protocol
for the new residual predictors rather than confusing replacement accuracy with
response fidelity.

For each of9 heads and512 sites, set pre'=pre−s*head_write, s=.1 or1; recompute RMS.
A same-head-span random control matches payload norm and dot product with pre.
One fixed control draw per head. Predict residual difference C(x')−C(x), compare
with [native(x')−compiled(x')]−[native(x)−compiled(x)]. Constant offsets cancel.
Conditional mediation starts from pre'+native(x), adds the compiled write change
and predicted residual change, then applies final RMS/vocabulary head. Target is
pre'+native(x'). This diagnostic uses a common native baseline, so it is NOT a
standalone replacement run. Predictors themselves have no native error access.
No fitting/selection occurs; same128 H259 documents, now with new interventions.

| Frozen predictor | Actual missing-response fidelity, scale.1 | scale1 | Conditional KL improvement over compiler, scale.1 | scale1 |
|---|---:|---:|---:|---:|
|Affine energy|.66%|.45%|−4.76%|−4.45%|
|Affine sensitivity|1.31%|1.13%|−2.27%|−2.47%|
|Quadratic energy|−8.35%|−8.64%|−5.60%|−6.37%|
|Quadratic sensitivity|5.29%|4.88%|5.64%|4.93%|

Response fidelity is1−squared prediction error/squared TRUE MISSING response,
pooled over documents and9 heads; compiler-only correction has zero fidelity.
Conditional KL improvement uses the compiler's response mediation as denominator,
not the natural-state baseline from H259. Therefore these percentages cannot be
subtracted from H259's53% natural-state KL recovery as a common-metric causal gap.
The comparable vector-energy story is also weak transfer: quadratic sensitivity
explained36.4% natural residual energy but only4.9% missing source-response energy.

**All three locked gates FAIL.** Quadratic sensitivity misses20% response fidelity,
10% conditional KL improvement, and5-point residual-fidelity advantage over affine
sensitivity at both scales. Nonzero benefits remain: full-scale response fidelity
4.88%, document-bootstrap95%[3.73,6.05]; KL improvement4.93%[2.02,7.46]; residual
advantage3.75 points[2.82,4.73]. Small-scale counterparts5.29%[3.99,6.67],5.64%
[3.09,8.05],3.99 points[3.00,5.01]. Intervals resample whole documents retaining all
heads and condition on fitted models/control draws. Failed gates are not zero effects.

Angle controls give quadratic-sensitivity residual fidelity4.01/3.25% at small/full
scale and conditional KL improvement7.83/7.17%. Thus poor response fidelity is not
confined to actual source directions or large deletion. Full-scale per-head actual
KL improvement ranges−12.7% to+10.6%; heads4 and5 worsen. All head results saved.
Pooled gain is not uniform circuit coverage.

**Downstream metrics disagree.** At full actual deletion, compiler-only logit-effect
fidelity88.64%, CE-effect fidelity59.11%. Adding quadratic sensitivity gives87.95%
and62.35%, respectively. It improves likely-token behavior somewhat while slightly
worsening uniform full-vocabulary logit reconstruction. The experiment does not
support a broadly accurate new causal mechanism. No top1 statistic is used to
inflate apparent success of small edits.

**Numerical/control audits.** Actual head-write replay relative error4.22e-7;
angle-control maximum relative norm discrepancy3.24e-7 and globally normalized
maximum pre-dot discrepancy2.55e-7. Small-scale conditional KL is about2.3e-6, so a
post-outcome precision audit recomputed softmax/log-softmax/KL in float64 from the
same float32 logits. Original files retained with _float32_kl suffix. Small-scale
actual improvement changes5.686→5.637%; full-scale4.928→4.928%; all conclusions
unchanged. Canonical results use audited KL. Oracle residual-change patch yields
logit/CE fidelity1 and maximum mean KL2.38e-13. This validates offsets and the
common-baseline path, not interpretation. Five relevant tests pass (patch algebra,
feature/solver checks and angle-control test); full suite not rerun. Plot inspected.
All H260 runs complete; original model and predictor weights unchanged.

**Interpretation.** Useful natural-state predictions do not imply faithful source-
response computation. The fixed feature map may miss interactions needed along
source changes, or natural-state training may leave those coefficients poorly
identified. Sensitivity-aware natural-state training helps modestly but does not
resolve that distinction. This strengthens T02/T08 and qualifies T10; it does not
prove features are noncausal, semantic, discrete, or impossible to learn.

**Next discriminator.** Keep the H259 feature maps and readout parameter counts
fixed; add explicit residual-response supervision from a predeclared subset of
heads on discovery documents. Hold other heads out of that supervision. Compare
natural-only versus joint natural/response fitting, with discovery validation,
frozen coefficients and test responses on seen and unseen heads plus angle controls.
If gains transfer to unseen heads, objective mismatch explains part of the failure;
if only supervised heads improve, apparent shared features remain source-limited;
if neither improves, this fixed feature class is a stronger suspect. Inspect earlier
L8 direct-objective experiments first; the new question is L17 residual-predictor
transfer, not a rerun of their four-coordinate result. Goal remains active.
