# M34: shared-dictionary comparison readiness

2026-10-28. Implementation and provenance audit, not a new training result.

## What was missing

M7 compared scalar, disjoint blocks and fixed overlapping groups on native MLP1 writes.
M8 explicitly retained only scalar/block models; later causal-supervision stages inherited
that omission. The independently developed SharedGroups module has learned gate encoders
and membership logits. It is not the same architecture as M7 shared_fixed.

The existing H687 synthetic results supplied true gates: oracle correctness and conditional
learning did not establish unsupervised group recovery. H836 used balanced memberships in
a different pre-MLP task-prediction setup. It repaired initial atom coverage, but learned
memberships still concentrated on a subset of atoms. Those results cannot be relabeled as
successful native-write feature discovery.

## Completed checks

The ten existing structure tests passed again, including union decoding, shared-atom removal,
multiple-parent closure, fixed-coordinate numerical gradients and a known computational DAG.
Hard-gate straight-through gradients are surrogate updates, not derivatives of discrete choice.

All six M7 saved models were independently decoded with NumPy on all 4096 evaluation positions
each (24,576 total). Explicit Python set unions exactly matched every production support.
Maximum normalized forward discrepancy was 7.10e-6. Saved GPU reconstructions replayed on CPU
within 1.35e-5 in normalized units; raw discrepancies reached .001587 because native writes
have a large scale. An initial raw absolute 1e-4 assertion failed; the audit now explicitly
checks normalized discrepancy, without claiming bitwise CPU/GPU equality or changing models.

| Implementation | Trainable parameters | Membership slots | Covered dictionary atoms | Active coordinates |
|---|---:|---:|---:|---|
| Scalar Top32 | 296,192 | N/A | 128 available | 32 |
| Disjoint Top8 blocks of4 | 296,192 | 128 | 128 | 32 |
| Fixed shared Top4 groups of8 | 296,192 | 256 | 128 | 28.26/28.44 means; 24–32 range |
| Separately encoded learned SharedGroups at these dimensions | 337,184 | 256 hard slots | Must measure after fitting | At most32 |

Scalar has no membership graph (the JSON records zero graph-covered atoms, not zero dictionary
capacity). Learned SharedGroups adds 36,896 gate-encoder parameters and 4,096 membership logits.
It uses sigmoid straight-through surrogates; M7 fixed models use energy-based hard gates and
selected-coordinate gradients. These are material implementation differences.

## Intervention semantics

For groups {a,b} and {b,c} active together, remove the first group gate and b remains selected
through the second group. Remove atoms {a,b}, and b disappears from every consumer. The audit
checks this distinction explicitly. Earlier four-contiguous-atom scalar/block interventions
must not silently be interpreted as shared-group gate deletion. Externally fixed native deltas
can still be applied identically across architectures for causal fidelity comparison.

## Fixed next comparison

The [design](validation/joint_shared_comparison_protocol.json) restores all six M7 starts to a
common 12-fit comparison: three existing architectures, two seeds, intact KL versus intact KL
plus native-response supervision. All start from the same reconstruction-training stage, rather
than comparing an M7 shared model against a much longer-trained M18 scalar model.

Reuse the fixed120-update natural/task schedule and ten competent task families. Each task
endpoint uses BOTH saved scalar and block native intervention generators for its seed, equally
weighted for every architecture. All240 endpoint assignments were checked against the existing
schedule, lexical split, template split and whole-task exclusions. One pooled training-only loss
calibration is computed across all six initial states before fitting; no new fits have run yet.
Gradient/interface checks and calibration are mandatory before launch.

This is equal trainable parameter count and a common active-coordinate ceiling, not a perfectly
matched active dimension, runtime or expressivity test. Random fixed sharing is a baseline; the
learned-membership alternative and manifold/reuse validation remain outstanding. Report actual
support costs and training-target retention. Previous response training failures make a negative
outcome plausible; retain it rather than tuning against the evaluation results.

M33 showed that native causal effects vary locally with context. The response objective already
uses the correct native target for each context; no invariance constraint is added. Separate
reconstruction, prediction, causal fidelity, structure validation and genuine task discovery.

## Reproduction

Run audit_shared_comparison_readiness.py in the established environment. It requires local M7
states and the native write cache, verifies their hashes, reruns the original structure tests,
and checks the existing schedule. [Audit record](validation/shared_comparison_readiness.json)
contains source hashes, all seed-specific costs and numerical discrepancies.

Historical references remain in the main research artifact directory: FEATURE_STRUCTURE_SYNTHETIC.md
and BILIN18_BALANCED_SHARED_FEATURIZERS.md. The broader goal is open; this audit establishes
readiness and a concrete comparison protocol, not superior shared features.
