

## H262 — Reversing supervised heads reproduces the transfer tradeoff

H261 made progress: fixed-feature response supervision learned trained sources but
harmed held-out sources. H262 reverses the split: train/validate response targets
on heads1/3/5/7, hold out0/2/4/6/8. All feature maps, natural documents, ridge grid,
beta0/1/10, fitting and evaluation protocols remain fixed. Reverse split has4
supervised heads versus prior5; losses average and energy-normalize examples, but
response target count differs. This is a qualitative reversal audit, not a clean
estimate of head-identity effect at exactly equal data count. No response targets
from held-out heads enter fitting/selection; ordinary natural states include them.

At full edits, quadratic beta0/1/10 supervised fidelity5.89/33.63/36.94%; held-out
4.72/−.59/−11.41%. Primary beta1 supervised gain27.74 points[25.91,29.60] and
conditional KL improvement19.17%[13.78,27.66]. Held-out fidelity drops5.31 points
[3.53,7.17] and conditional KL worsens4.20%[1.34,7.01]. The supervised-learning gate
passes; held-out-vector and behavioral-transfer gates fail, as in the original split.
Small edits give the same gate outcomes. Quadratic natural KL recovery42.17→40.37→
16.13% as beta increases; affine37.35→30.59→−3.97%. Strong response weighting is
particularly costly here. Failed transfer is not unique to the original odd heads.

## H263 — Direction diversity changes the tradeoff but does not rescue transfer

Freeze two additional training/validation populations before their outcomes:
(1) random directions in each original even head's output span; (2) random directions
in full residual space. Both match each actual even-head write's norm AND dot product
with pre. Thus equal edit size and RMS geometry are preserved. The second is named
'isotropic' in filenames but is an AMBIENT ANGLE-MATCHED distribution, not an
unconstrained isotropic Gaussian; its parallel component is fixed. Same5 norm donors,
number of training edits, feature maps, ridge/beta grid and evaluation directions.
Losses normalize each target population's energy, not identical absolute target
energy. No native head-ID input. For ambient training, 'supervised' labels in JSON
refer to norm donors; training directions are not restricted to their head spans.

Primary comparison: quadratic beta1, actual full-edit evaluation. H261 actual
training is the frozen reference. No best variant selected on these test results.

| Training response directions | Even-head fidelity | Odd-head fidelity | Natural KL recovery |
|---|---:|---:|---:|
|No response training (beta0)|4.72%|5.89%|42.17%|
|Actual even-head writes|38.04%|−.85%|44.90%|
|Same-head angle-matched random|6.58%|−3.39%|44.20%|
|Ambient angle-matched random|6.13%|7.23%|34.78%|

Both locked >=10-point held-out-fidelity rescue gates FAIL. Same-head random
training worsens odd-head fidelity by2.54 points[1.23,3.87] relative to actual
training. Ambient random training improves odd-head fidelity8.07 points[6.79,9.44],
below10, while losing31.91 points[29.13,34.81] on even-head actual edits. At small
scale ambient gain7.83 points[6.53,9.21], again below10. Direction coverage therefore
matters without producing a high-fidelity shared response rule.

The apparent ambient gain is mostly avoidance of actual-training harm: versus its
own beta0, ambient beta1 improves odd-head fidelity only1.33 points[1.01,1.65].
Its conditional KL versus beta0 worsens1.58%, with improvement interval[−2.70,.18]%
including zero; versus actual training it improves2.33%[−.10,5.32]%, also uncertain.
Hence there is no demonstrated downstream rescue at full scale. Natural KL recovery
also falls42.17→34.78%. At beta10 ambient even/odd fidelity3.68/3.73% and natural
KL recovery17.29%; greater response weight does not solve this. All beta/affine,
angle-test and per-head outcomes are retained, including unfavorable ones.

**Audits and scope.** This turn fits54 predeclared candidates across reverse,
same-head random and ambient random conditions, with discovery-only selection.
Same128 H259 evaluation documents are reused, not a fresh replication. Paired
bootstrap2000 resamples whole documents retaining heads; intervals condition on
models and random training/evaluation draws. Direction-control max relative norm
error<=3.41e-7, globally normalized pre-dot discrepancy<=1.88e-7. Coverage-fit double
normal-equation relative residual<=2.42e-15. Six directly relevant tests pass;
full project suite not rerun. Figure inspected. All H262/H263 runs complete; original
model unchanged. Shared H261 training runner now accepts training_payload and seed;
default actual-input behavior is preserved. Original outputs remain intact.

**Scientific update.** Source-specific fitting tradeoffs persist when the head split
is reversed, and random directional coverage does not remove them. This weakens
'simply change the loss' and 'simply add arbitrary directions' accounts within this
fixed feature class. It raises the priority of representation/feature-map limits,
but does not prove them: joint-loss regularization, finite discovery data and the
specific direction samplers remain possible contributors. Actual source-context
statistics are useful for targeted prediction, yet learning them here does not
identify a transferable fundamental unit. Do not equate head identity with a unit.

**Next capacity discriminator.** Fit responses from ALL9 heads with the same feature
maps and separately inspect training, validation and test response fidelity. Include
response-only fits (intercept fixed by natural mean) to remove the natural-loss
tradeoff, with ridge selected on discovery response validation. If even training
response fidelity remains low at negligible ridge with a verified solve, the fixed
function class is directly implicated on that sample; if train is high but new
contexts fail, estimation/generalization is implicated. This changes supervision
coverage deliberately and is not an unseen-head transfer claim. Then vary the
quadratic input subspace/rank or use tensor-derived interaction partners at explicit
cost, rather than repeatedly tuning losses within a demonstrably limited map.
Broad research goal remains active.
