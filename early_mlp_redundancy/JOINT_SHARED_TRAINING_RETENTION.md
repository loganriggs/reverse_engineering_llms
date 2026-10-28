# M35–M36: common scalar/block/shared training and final target retention

2026-10-28. Continues [M34](SHARED_COMPARISON_READINESS.md).

All12 common-training fits completed and their final states/schedules passed audit. Native-response supervision improves final training response error versus matched intact-KL control in 6/6 starts. These are familiar training targets; held-out native response and ordinary-output evaluation remain pending. Fixed random sharing is a baseline, not discovered reusable structure.

## Fair comparison and calibration

Three existing M7 architectures, seeds71/72, start from the same reconstruction-training stage.
Each has296192 trainable parameters and128 atoms. Scalar/block use32 active coordinates;
fixed-overlap groups select at most32, with fewer when selected memberships overlap.
All models use the same seed-specific natural/task schedule,120updates, Adam .0003 and unit
column decoder normalization. Both arms are retained: intact native-output KL and KL plus
native-response loss. This matches schedules and parameter counts, not active dimension or FLOPs.

Each task endpoint has both previously frozen scalar-generated and block-generated native
perturbations, averaged equally for every architecture. Native context-specific responses are
targets; no output invariance across wrappers is imposed. Noops remain in the loss/averages.
The response loss compares each reconstructed model's perturbed-versus-own-intact response
with the native response, differentiating both reconstructed branches. Native weights and
perturbation generators remain frozen.

Before training, all720 scheduled endpoints across six starts were evaluated (1989forwards).
One common loss weight, pooled meanK/pooled meanR, was fixed at7.702620877869684. This equalizes
pooled initial loss magnitudes, not gradients or per-model difficulty. It uses training data only.
First scheduled prefix/task checks for all six models passed24 reconstructed-write directional
finite differences (maximum relative discrepancy0.149%), native-replacement identity checks,
both-branch derivatives, and finite selected-support parameter gradients. This does not validate
derivatives across discrete support changes or guarantee trainability.

An initial metric audit failed only on float32 CPU/GPU active-count mean rounding (~2e-6).
It now compares to double-precision counts with5e-6 tolerance; all loss calculations and model
parameters were unchanged. Independent NumPy sampled loss discrepancies were below3.89e-16.

## Final training-target scores

All12 fits finished120updates; all1440 update records, target assignments, model hashes,
fixed memberships and decoder normalization were audited. Every final model was then rescored
on all120 scheduled training endpoints and both generators, regardless of its training arm.
This uses3978 further forwards, not online minibatch losses.

| Kind / seed | Initial R | KL-only final R | Joint final R | KL-only final K | Joint final K |
|---|---:|---:|---:|---:|---:|
| scalar 71 | 0.011415 | 0.002798 | 0.002640 | 0.022887 | 0.031153 |
| block 71 | 0.016999 | 0.004891 | 0.002910 | 0.024113 | 0.036757 |
| shared_fixed 71 | 0.016493 | 0.008353 | 0.003741 | 0.031196 | 0.043353 |
| scalar 72 | 0.017972 | 0.006705 | 0.005477 | 0.020851 | 0.040582 |
| block 72 | 0.027695 | 0.006834 | 0.006557 | 0.025716 | 0.045996 |
| shared_fixed 72 | 0.045482 | 0.023765 | 0.005239 | 0.026331 | 0.073836 |

Causal supervision improves R in all six paired comparisons, while worsening intact K relative
to the KL-only control in all six. Both objectives improve relative to their M7 initial states.
The shared models reduce R by55.2% and78.0% versus their controls; their intact K rises39.0%
and180.4% respectively. This is a causal-fidelity/ordinary-prediction tradeoff on training data,
not a demonstration of shared architecture superiority. Unlike M20's failed retention, the
common comparison starts earlier and changes generator coverage and loss calibration as well;
it does not isolate which change enabled retention.

K is mean intact-output KL over all positions; R is native-probability-weighted squared error
in final-token log-probability intervention responses, averaged over both generators. These
metrics are different units. Absolute scores may differ by seed because interventions and
schedules differ; compare paired arms first. Detailed family means and actual support costs
are included in the results. No winning checkpoint or architecture is selected from this table.

## What this does and does not resolve

The shared baseline now receives the same causal-supervision opportunity and training history
as the scalar/block controls. This closes its omission from the previous causal-training line.
It does not establish learned group discovery, semantic manifolds, selective reuse or an optimal
architecture. The learned-membership implementation remains distinct and untested in this
specific common native-write training setup.

NEXT: evaluate every final model and its M7 initial state on the fixed native perturbation
benchmark and full ordinary-output inventory. Preserve source-generator labels, zeros,
competence failures, families and lexical/template quadrants, plus natural-text forgetting.
Training retention is necessary evidence but cannot substitute for that transfer evaluation.
Do not start another optimizer sweep or discard failed arms. Then use actual transfer and
support coverage to decide the next shared-membership/structure validation step.

All data remain familiar development data. Preserve sequence_copy, spatial_inverse,
weekday_successor, role_binding and tense as whole-task training exclusions; past_tense is a
separate allowed family. No pristine holdout or new-task discovery claims.

## Evidence and reproduction

- [Frozen comparison design](validation/joint_shared_comparison_protocol.json)
- [Common calibration](validation/joint_shared_calibration.json), [gradient checks](validation/joint_shared_gradient_checks.json), [calibration audit](validation/joint_shared_calibration_audit.json)
- [Training inventory](validation/joint_shared_training_results.json), [training audit](validation/joint_shared_training_audit.json)
- [Final retention](validation/joint_shared_retention.json), [retention audit](validation/joint_shared_retention_audit.json)

Scripts: joint_shared_loss.py, calibrate_joint_shared.py, check_joint_shared_gradients.py,
audit_joint_shared_calibration.py, train_joint_shared_comparison.py, audit_joint_shared_training.py,
evaluate_joint_shared_retention.py and audit_joint_shared_retention.py. Use the established pinned
model environment with single-thread BLAS/OpenMP. Model/checkpoint/vector caches remain local
at hash-linked paths in the JSON; the small shareable packet carries scripts and scalar records.

Audit scope: full scheduled-row/aggregate bookkeeping and independent NumPy scoring of24 sampled
final outputs and48 intervention responses, not independent replay of every forward or optimizer
step. Model native identity and gradient checks provide additional implementation evidence.
