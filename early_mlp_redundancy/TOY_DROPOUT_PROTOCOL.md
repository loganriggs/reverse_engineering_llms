# T4 fixed protocol: jointly train input and channel robustness

Frozen before outcomes,2026-10-28. Fine16-vector target, token noise.25, contribution penalty.01,
width16/32, channel dropout0/.25, three seeds0/1/2. Reuse three T3 width16/no-dropout models;
train nine new models. Same Adam.01,2000steps,batch256, Gaussian initialization std.2. No tuning.
Same underlying input generator per seed; dropout uses an independent generator so it does not
change data streams. Fresh uniform fixed-size mask shared across the batch, deleting exactly25%
of product channels. No inverted scaling. Output bias remains. Contribution penalty is measured
on unmasked contributions in both arms; task loss uses the masked forward only in dropout arms.
All channels used at evaluation. Different widths have different parameter counts and initial
function distributions; this is not a pure factorization-equivalent initialization comparison.

Evaluate identical balanced1024-example independent draws at token noise0,.25,.5; sixteen fixed
25%-deletion masks on clean inputs; every single-channel deletion on clean inputs. Report absolute
and intact-relative vector error, nearest-target accuracy/confusions, global and per-class worst
single-channel accuracy, channel/net energy, participation and class specificity. Worst channels
can differ across classes, so that statistic is not one global intervention. Unrescaled dropout
may alter intact output calibration; do not repair that post hoc or hide the clean-vector cost.

Archive all runs and states. Stop the toy grid after this fixed comparison and map its signatures
to real-model evidence/gaps. No claim about held-out identities/combinations, label noise, LLM
training causes or a universal robustness frontier. Reused controls are already known outcomes.
