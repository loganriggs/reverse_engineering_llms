# Latest priority: capacity and robustness of learned implementations

The user's extremes question is now tested in [T2](TOY_CAPACITY_RESULTS.md), with a
[pre-outcome protocol](TOY_CAPACITY_PROTOCOL.md). Next hold the fine target task fixed and compare
width8 with16, retaining paired seeds, penalties, noise levels and training budget. This isolates
capacity more directly than changing both target complexity and capacity demand. Label-noise and
held-out-combination tests remain unperformed. The broader taxonomy and real-layer evidence below
remain relevant; do not equate toy results with an explanation of the checkpoint's training.

## Broader research direction: identify computational mechanisms

The user's latest request broadens the question from removal tolerance to how MLPs implement
functions, including clustering, overlap and cancellation. Use [COMPUTATION_STRATEGIES.md](COMPUTATION_STRATEGIES.md)
for the operational taxonomy, twenty hypotheses, first exact-function toy and fixed next sequence.
Do not treat these as mutually exclusive layer categories. The [matched comparison](MATCHED_LAYER_MECHANISMS.md) is complete and rejects a simple layer
dichotomy on this panel. Next add shared-plus-private contributions and controlled downstream
readers to the diagnostic toys before training explanatory models. Corpus checks remain approved
and necessary for generalization; they are not the sole next step.

## Preserved subquestion: removability of MLP1

The user's definition is: can many parts be removed while the model performs the same computation?
MLP1 is zero-based block1. Separate four claims: a clustered signal can be patched; a module can
be approximated by a token table; a direction can steer behavior; a subset can be deleted without
materially changing outputs. Only the last directly addresses removal redundancy.

Read [REMOVAL_REDUNDANCY.md](REMOVAL_REDUNDANCY.md) for the source-level history correction,
fixed literal-deletion experiment, limitations and next comparison. Read the README for loading.

The primary outcome is native-versus-edited output distribution, supplemented by task accuracy,
observed-token CE, worst-case examples and local write changes. A preserved binary decision alone
is insufficient. A distributional tolerance must be explicit before making an approximate-equivalence
claim; this diagnostic reports effects without inventing a passing tolerance after the fact.

The fixed [direction-preserving norm control](DELETION_NORM_CONTROL.md) is complete: at matched
magnitude it causes substantially less KL than literal deletion. Next audit natural-text row
provenance and freeze a finite corpus evaluation of the same masks and controls. Do not optimize
masks or call reused/unknown rows fresh. Preserve per-document tails and position-stratified results.
The task-panel results remain descriptive until this generalization boundary is tested.

The previous [context-response plan](ARCHIVED_CONTEXT_RESPONSE_PLAN.md) is retained as history.
Its proposals and P10's projection/residual test are deferred, not additional active next steps.
The broader goal remains to identify faithful computational units and eventually use diverse
competent-task causal targets to improve featurizers, preserving whole-task evaluation exclusions.
