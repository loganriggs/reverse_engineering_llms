# Preregistration: same-codebook initialization invariance

The different-manifest leave-one-relation-out test was non-adjudicative because
the source and target models assigned independently randomized output labels to
the omitted relation.  This factorial holds the manifest, grammar, examples, and
answer permutation fixed while varying only initialization and training order.

Three depth-4 bilinear models passed the behavioral fidelity gate before any
representation result was inspected.  A fourth initialization that failed
held-out-template transfer is excluded by that pre-existing gate and retained as
an optimization-failure record.

For each ordered initialization pair and direct relation, fit the same global
ridge activation map on ordinary prompts from the other seven relations.  Freeze
it, evaluate state reconstruction on the omitted relation/template 6, and map a
semantic edit for that relation.  Controls and metrics are identical to the
different-codebook red team.

This separates two hypotheses:

- **algorithmic invariance:** independently trained solutions share a linearly
  alignable computation once the task codebook is fixed;
- **codebook-relative organization:** random label conventions determine the
  apparent feature coordinates, so unseen conventions cannot be aligned without
  calibration.

Even a positive result establishes only an affine equivalence class, not matched
neurons or a unique canonical basis.
