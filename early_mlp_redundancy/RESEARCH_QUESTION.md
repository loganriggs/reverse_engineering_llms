# Active next step: validate candidate block structure in frozen models

[M11](DIVERSE_CAUSAL_SUPERVISION.md) completes8fixedfits and source-separatedtransfer/forgettingevaluation. Freeze finalmodels. Audit conditionalblockdimensionality, occupancy, seedsubspace/supportstability and selectivecausalroles withscalar/shuffledgroupcontrols. Handle sign/permutation/withinblockrotation. Choosecandidatesonlyfromtrainingdata; do notrepeatedlyselectontiny14pairevaluation. Stable/multidimensional≠semanticmanifold. Originalwhole-taskexclusionsreserved; stopmixture/width/gate tuning. Earlierplansarehistorical.

# Completed step: execute frozen mixed-supervision comparison

[M10](DIVERSE_SUPERVISION_SPLIT.md) completes metadata splitting and independent audit. Execute exact `validation/diverse_supervision_protocol.json`: fourM8causalinitialstates×natural-only/mixed arms,120Adam.0003updates, frozenfamilybalancedtask/naturaldocument schedules, noselection. Evaluate14pairs/6families,66trainingpairs separately andnaturaldevelopmentdocs32–63. Fourfamiliesaretraining-only; no lexicaltransferestimate. Comparefrozenbaseline andbotharms,retainallfailures,originalexcludedfamiliesuntouched. Do notresplitafteroutcomes. Earlierplansarehistorical.

# Completed step: audit splits for diverse causal supervision

[M9](FEATURIZER_TASK_TRANSFER.md) finds average transfer but consistent family-specific failures. Audit lexical source-token groups/context duplicates in the existing10trainingfamilies, then freeze a bounded family-balanced taskKL versus continued natural-textKL comparison fromidenticalcheckpoints. Keep paired endpoints/lexicalgroups together fortransferclaims; flagfamilies lacking multipleindependentgroups. Retain natural-text eval to detectforgetting, and original excludedfamilies. Do not fit selected failures then call them validation. No manifold/newtaskclaim or additionalnatural-textobjective search. Earlier plans are historical.

# Completed step: frozen diverse-task transfer

[M8](CAUSAL_FEATURIZER_OBJECTIVE.md) finds59–63%lower outputKL under explicit downstream training than continuedMSE, with worse rawreconstruction. Freeze all8models. Evaluate existing80-pair/ten-family task panel without fitting, retaining allseeds/families and original exclusions. Measure outputKL/top1, supported task decisions/margins and localerror separately. These tasks were not featurizer training data but were previously examined in thisresearch; do not call them pristine holdouts or discoveredtasks. Stop natural-text objective tuning; test transfer before interpreting groups. Earlier plans are historical.

# Completed step: explicit causal objective versus continued reconstruction

[M7](WRITE_FEATURIZER_PILOT.md) completes six matched-parameter fits and live causal evaluation. Freeze a bounded fine-tuning comparison from the saved scalar/block checkpoints: continued reconstruction versus native-output KL supervision, matched train-document streams and optimizer steps, no evaluation-based selection. Verify differentiable native replay/gradient path first and fix loss scaling on training data only. Preserve active32 budget, evaluate reconstruction/token/context/KL/CE separately, retain both seeds, and keep original task exclusions unused. Do not tune fixed-shared gates or widen models based on this pilot. Earlier plans below are historical.

# Completed step: reconnect the causal benchmark to featurizer fitting

[M6](MATCHED_ANGLE_CONTROL.md) completes the fixed directional controls. Stop perturbation/seed sweeps. Audit existing scalar/block/shared implementations and prior failures (H835 pre-MLP sites; H984/H989 MLP12; H1029 stopped selector). Prepare a bounded native MLP1-write comparison with document-separated fitting, matched active-coordinate costs and causal output evaluation, preserving raw/token/context reconstruction separately. Fixed shared memberships are a baseline; do not resume collapsed learned gates without a new reason. Freeze protocol before training. This is a benchmark and representation comparison, not proof of manifold semantics or new-task discovery. Earlier plans below are historical.

# Completed step: matched-angle directional control at MLP1

[M5](TOKEN_TABLE_CAUSAL.md) shows table replacement has substantially greater downstream effects than norm-only changes. Before attributing this to special contextual directions, match per-position final product norm AND angle from native write using random tangent directions at MLP1, same covered positions/bias/cohort, fixed small seed set and all results. Freeze construction and seeds before outcomes. This compares equal local errors with different orientations, not semantic features. Stop after the fixed control; use outcome to choose causal-featurizer targets. No whole-task holdout or toy-grid expansion. Prior next-step sections are historical.

# Completed step: causal test of cross-document token-mean predictions

[M4](NATURAL_TOKEN_CONTEXT.md) shows much larger contextual variation on natural text and limited MLP2 token-mean prediction. Use the same64 reused training documents and opposite-half token means; replace product writes at covered positions, retain Down bias/native uncovered writes, and test each early layer separately. Include native-product norm-only scaling to table norm, output KL/top1/CE, coverage and local vector error. No refit, new task holdout, or hyperparameter sweep. Freeze this protocol before outcomes. These are table-replacement interventions, not evidence of native-channel removability. Earlier next-step sections are historical.

# Completed step: natural-text replication with audited provenance

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
