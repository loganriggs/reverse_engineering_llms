# Active next step: natural-text replication with audited provenance

[M3](TOKEN_CONTEXT_PARTITION.md) completes the token/context partition and independent audit. Audit existing natural-text data and cached native writes before freezing a finite replication of the same decomposition. Verify model/site and token-prefix alignment, document prior use, and label reused data honestly. Preserve whole-task exclusions. The featurizer deliverable is separate token/context reconstruction diagnostics plus causal fidelity, conditional on broader replication; this observational partition does not establish causal features. No additional toy width/noise/dropout tuning.

# Completed step: token geometry versus contextual variation

[M2](EMPIRICAL_OUTPUT_SPECTRA.md) completes the real-output spectral mapping. Next decompose
observed writes into token-conditioned means and within-token residuals on the same opened panel,
with occurrence and unique-prefix weighting, singleton counts and exact energy accounting.
Do not interpret within-sample means as a validated predictor or context removal as a measured
causal intervention. This separates potential clustering/re-clustering from contextual computation
before choosing a featurizer target. Prior next-step paragraphs below are historical.

## Completed step: map the completed toy grid to real-model signatures

[T4](TOY_DROPOUT_RESULTS.md) completes the fixed capacity/dropout comparison and closes that toy
grid. Read [STRATEGY_SYNTHESIS.md](STRATEGY_SYNTHESIS.md) for the current mechanism account.
Next measure empirical output spectra and contribution-energy bounds for MLP0/1/2 on the same
opened inputs, including deduplicated causal-prefix views. No fit/pruning or claim that a matrix
rank bound yields an executable compact bilinear MLP. Preserve whole-task exclusions and distinguish
this task-panel diagnostic from natural-text generalization. Prior next-step paragraphs below
are historical, not simultaneous experiment queues.

## Completed step: can both toy robustness types be trained?

[T3](TOY_WIDTH_RESULTS.md) completes the fixed width comparison and checks class specialization
and selective channel vulnerability. Its next-test section freezes the small dropout/capacity
comparison: fine targets, input noise.25, penalty.01, widths16/32, channel dropout0/.25, three seeds,
fixed budget. Reuse three existing models and train nine new ones. Stop this toy grid after that
comparison and relate the diagnostic signatures to the real-model evidence, without assuming a
universal tradeoff. Prior plans below are preserved history, not additional simultaneous next runs.

## Previous step: capacity and robustness of learned implementations

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
