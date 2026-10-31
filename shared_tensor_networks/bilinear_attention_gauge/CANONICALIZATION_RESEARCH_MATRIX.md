# Canonical reduction and legibility: experiment matrix

## Question and objects

There are two choices to make independently. The **reduction metric** says which
changes to the attention function count as expensive. The **legibility prior**
says which coordinates or modules we prefer among sufficiently faithful
reductions. Weighted trace gauge supplies reproducible coordinates for one
specified network and metric, but cannot alone select a unique decomposition
of the function across QK branches or heads.

The first-layer object is the exact multi-head query/key/output response,
including both QK paths, RoPE, and OV. For one-hot tokens this can be studied
without sampled sequences. For the 546M model a dense full-vocabulary
query–key–output tensor is prohibitive, but the factors support implicit
contractions and Monte Carlo estimates with explicit coverage/error checks.
Sequence weighting is a **different objective**, not a computational
requirement for defining attention.

For a declared measure `mu`, fit a reduced function `F_r` by minimizing
`E_mu ||F-F_r||²` at each rank/resource budget. In a second pass, optimize a
legibility cost `L(F_r, coordinates)` while bounding the increase in that
same function error. Exact gauge rotations make the bound zero. Any claimed
gain must then survive the *other* measures and held-out interventions; the
training objective is not itself the validation criterion.

## Stage 1: choose what reduction preserves

| Metric | What it optimizes | Necessary challenge set |
|---|---|---|
| Uniform token pairs | Function on a broad token domain, including rare tokens | Independent uniform pairs; rare-token and position-gap strata |
| Natural corpus pairs | Expected response on observed text | Disjoint corpora, documents, and topic/grammar shifts |
| Floored or mixed measure | Trade broad coverage against natural usage | Sweep floor/mixing weight; report both uniform and natural error |
| Branch-intervention ensemble | Preserve QK1/QK2 edits as well as native response | Hold out edit strengths, compositions, and edit families |
| Downstream logits or KL | Preserve final model behavior | Separate pair/head-output fidelity from final KL and task accuracy |

The uniform metric is data-free in the sense of needing only weights and a
declared token domain. Corpus and downstream metrics depend on a data
distribution. For each rank and parameter budget, report the complete
cross-metric Pareto frontier rather than naming one metric universally best.

## Stage 2: choose what makes components legible

Evaluate these at a **fixed reduced function first** using exact gauge changes
when possible; then allow the reduced function to change within a stated error
budget. This isolates coordinate choice from lost information.

| Prior or objective | Intended structure | Main failure check |
|---|---|---|
| Native weight sparsity | Few physical coefficients | Exact gauge and QK-branch-swap sensitivity |
| Canonical-bond sparsity | Sparse coordinates after bond gauge fixing | Seed stability; cross-head redistribution |
| Token or pair-response sparsity | Each axis affects a compact set of token pairs | Frequency-matched controls and held-out token forms |
| Group sparsity over shared dictionary | Reused subspaces with selective gates | Whole-task holdouts, group-count sensitivity, independent starts |
| Intervention-response sparsity | Axes with selective causal effect profiles | Unseen edits, dose matching, downstream propagation |
| Computation or description-length penalty | Few reusable nodes/edges across heads | Same-function alternative factorizations and graph recovery |

No sparsity score is itself an interpretability score. Report reconstruction,
task prediction, causal fidelity, axis/group stability, known-rule recovery,
and discovery of withheld rules separately. Compare priors at matched
reconstruction error, parameter count, rank, and intervention dose. Model
selection may see training tasks and validation edits; discovery claims require
whole withheld task families and fresh test interventions.

## Testbed ladder and decision rule

The next comparison should be a small factorial, with **reduction fitted
before legibility optimization**. At a matched rank and function-error
budget, cross three fitting measures (uniform weight-only token domain,
natural corpus, and a mixture with an explicit uniform/rare-edit floor)
with four coordinate objectives (weight sparsity, token-response sparsity,
intervention-response concentration, and sparse reusable conditional
products). Include the unrotated SVD basis, exact random rotations, and
full-head response norm as controls. For the first pass, use exact rotations
inside a fixed reduced subspace so every objective has *identical* reduced
function fidelity; only then let the subspace itself move under a stated
error budget. Plot broad-domain error, natural-domain error, held-out
finite-intervention error, and legibility/rule recovery as separate axes.

Use a two-head synthetic attention function with a common rule, a rare
rule, a shared computation reused by both heads, and a duplicated-function
alternative to test identifiability. Vary the rare-rule probability and
noise in query/key tokens; evaluate enumerated token pairs as the exact
weight-only reference. Then train one- and two-layer toy-language models
with the same rules and with held-out rule combinations. Finally test
predeclared winners and failures in the 546M model, including all-head
scans so a head is not chosen with the target label. Require independent
seeds and fresh edit types or task families before calling a recovered
coordinate a discovered mechanism.

1. **Hand-built attention controls:** two heads with common and rare rules,
   shared versus duplicated computation, branch swapping, and exact token
   function. These prove algebraic invariances and expose obvious biases but
   cannot demonstrate discovery in a trained network.
2. **Trained toy language:** one- and two-layer bilinear attention models with
   multiple nonredundant outputs, rare compositional combinations, variable
   token counts, and at least one context-dependent interaction. Hold out
   complete combinations and intervention types. Recover known rules without
   using their labels for unsupervised fitting or selection.
3. **Small real checkpoints:** vocabulary-small models where all token pairs
   can be enumerated, to benchmark sampling and distinguish approximation
   artifacts from genuine metric tradeoffs.
4. **546M model:** first-layer all-head dictionary, then full 18-layer
   propagation on FineWeb, WikiText-103, TinyStories, rare-token strata, and
   edited inputs. Measure whether proposed components predict selective
   behavior better than random, frequency-matched, and same-gauge controls.

Advance a prior only if it retains fidelity on the declared broad and natural
domains, improves rule recovery on trained toys, and makes a prediction about
new inputs/interventions that survives controls in the real model. If local
pair labels fail in the full model, report the failure rather than optimizing
the same label on the test documents.

## What the existing experiments already show

The [factorial toy](FACTORIAL_PLAN_AND_TOY_RESULTS.md) shows that uniform and
usage-weighted rank-one cuts retain different common/rare rules; the prior
sweep finds a real fidelity–intervention tradeoff, not a free gain. The
[causal-decoy two-head control](CAUSAL_DECOY_TOY_RESULTS.md) now explicitly
separates the two axes of the matrix. Exact uniform pairs favor the rare
rule, usage pairs favor frequent coactivation, and a uniform floor trades
between them. Hidden-response rank-one reduction selects a large decoy
with almost no output value. At an exact full-function gauge, ordinary
pair/output-response L1 favors mixed coactivation axes, while a designed
edit-response prior recovers the two ground-truth rule axes in three
physical gauges. The edit ensemble uses known input structure, so this
is a control, not blind discovery; trained toy-language and fresh-edit
transfer remain the required gate. The
[trained necessary-two-head retrieval factorial](DUAL_JOINT_EDIT_GAUGE_RESULTS.md)
crosses uniform, empirical, and mixed rank-two reductions with key, output,
component-response, whole-sequence response, all-pair edit, and role-aware
edit gauges on ten independently trained models. All rank-two functions keep
full and rare accuracy; label-free all-pair edit sparsity recovers the two
known key variables about as well as the role-aware control. Crucially,
whole-sequence response L1 instead selects coactivation mixtures and makes
reduced-function component deletions less selective, especially under a
synthetic correlated usage law. A
[correlated-training follow-up](CORRELATED_DUAL_JOINT_GAUGE_RESULTS.md)
confirms the conflict in ten freshly trained models under a strongly
correlated law. Its unlabeled frequency-weighted rank-two cut also repairs
all four full-model failures on the withheld rare combination, while
moving further from the original function than a uniform cut. This is a
task-versus-faithfulness tradeoff, not a better reconstruction. The
[trained toy sequence](THREE_COUNT_CANONICAL_RESULTS.md) shows rank competition
and differing uniform, frequency, and supervised-CE outcomes. The
[nonlinear-readout follow-up](THREE_COUNT_NONLINEAR_METRIC_RESULTS.md) shows
that optimizing a downstream bilinear output at rank two can raise overall
accuracy while eliminating a withheld rare-combination behavior across ten
trained attention checkpoints. Its
[fixed-function gauge follow-up](THREE_COUNT_NONLINEAR_GAUGE_RESULTS.md)
shows that key sparsity and context-edit sparsity select very different
coordinates for the same function: key sparsity recovers A/B key variables,
while context-edit sparsity rotates them into mixtures to make single-axis
effects look additive. A
[trained quadratic-readout control](THREE_COUNT_TRAINED_COMPOSITION_RESULTS.md)
replicates this conflict across ten learned compositions. A
[from-scratch control](THREE_COUNT_SCRATCH_COMPOSITION_RESULTS.md) confirms
it in ten independent end-to-end-trained models and tests a weight-only
quadratic diagonalization gauge. A [reuse-aware graph-prior sweep](THREE_COUNT_GRAPH_PRIOR_RESULTS.md)
on those same exact reduced functions trades key sparsity against the number
of quadratic product nodes: a moderate penalty preserves most designed-variable
alignment and reduces held-out one-node pruning error. This remains a toy,
post-sweep result; it does not recover the physical circuit. A
[fresh-seed test](THREE_COUNT_GRAPH_PRIOR_FRESH_RESULTS.md) finds the
uniform-cut pruning gain only partly transfers and the frequency-cut
gain mostly disappears; the coefficient-count surrogate is not yet a
reliable graph prior. A [direct post-pruning error-budget comparison](THREE_COUNT_PRUNING_BUDGET_RESULTS.md)
replaces that proxy with a constrained function measure. Complete uniform
token enumeration and unlabeled context fitting select different exact
gauges and transfer differently; neither repairs task information already
lost in the rank-two cut. The
[shared-pair task-faithful control](SHARED_PAIR_CANONICAL_RESULTS.md) trains ten
new models with two related parity outputs. Rank three preserves all five
outputs; a label-free key-sparse gauge exposes three conditional product
edges and their query-specific reuse. Keeping only those three nodes retains
the held-out task, whereas the SVD gauge loses accuracy after matched
pruning. This establishes function-level reuse in a new toy, not physical
head/node identity. The
[dense-token red team](DENSE_CODED_SHARED_PAIR_RESULTS.md) changes the
vocabulary coding so every token contributes to all three variables.
Key-entry L1 then fails, while condition-wise monomial sparsity and a
constructive joint-nullspace gauge recover the reusable graph and retain
the rare holdout after three-node pruning. This isolates a stronger
functional prior than token sparsity, but still does not identify the
physical attention implementation. The
[trained dense-code edit-gauge test](SHARED_PAIR_DENSE_EDIT_GAUGE_RESULTS.md)
adds a different label-free prior: L1 on all 28 token-swap responses
within the fixed rank-three function. Across ten trained seeds and both
uniform and frequency cuts, it nearly perfectly recovers the three
known variables and retains 100% five-output accuracy after selecting
three quadratic nodes, including the rare whole-combination holdout.
It does not need hand-chosen Hamming-one edges, but exploits the
complete eight-token cube geometry. A three-model random-subset sweep
shows that 24 distinct swaps succeed in 89/90 panels, whereas 56 draws
with replacement (24 distinct on average) succeed in 61/90. This
strengthens the case for intervention diversity and controlled weights
as gauge priors; physical QK-branch faithfulness remains unsolved. The
[synthetic edit-noise red team](SHARED_PAIR_DENSE_EDIT_NOISE_RESULTS.md)
shows a second constraint: exact all-pair recovery is fragile to noisy
edit-vector estimates. At noise standard deviation .2 times median
edit norm, one observation per pair succeeds in 2/30 trials, whereas
averaging 16 independent observations succeeds in 30/30; a persistent
pair-specific bias remains at 2/30 regardless of repetitions. At
higher noise, 64–256 independent repeats are needed. This is an
observation-noise control, not a claim about measured 546M activations.
The [second trained composition task](THREE_COUNT_SCRATCH_EDIT_GAUGE_RESULTS.md)
now transfers the all-pair edit prior to ten from-scratch single-parity
models. Under both weight-only and unlabeled-frequency rank-three cuts,
it recovers the known variables and keeps 100% full and rare-holdout
accuracy after matched two-node pruning. Key-entry L1 is equally good
on this paired-token coding, so the result strengthens transfer across
related tasks rather than establishing a universal advantage. Broader
token codings, context-dependent tasks, and physical-branch tests remain
necessary.
The [edit-family holdout and coverage sweep](THREE_COUNT_EDIT_FAMILY_HOLDOUT_RESULTS.md)
withholds all four direct within-pair bit flips from gauge fitting. Across
ten trained three-variable models and five connected edit-graph orders,
fourteen cross-pair swaps suffice for all 100 uniform/frequency fits to
recover the variable axes above .999 cosine and preserve the full and rare
task after two-node pruning. Seven edits already retain the task in every
fit but leave some variable axes unstable. Because the edit graph is
connected and all tokens are seen, this tests orientation transfer to
withheld edges, not prediction of unseen-token effects.
The [finite-input-edit transfer](THREE_COUNT_FINITE_EDIT_TRANSFER_RESULTS.md)
then evaluates 51,200 actual before/after substitutions in the trained
models on fresh ordinary and higher-count contexts. The exact token table
passes a direct-logit check below `2.1e-12`; rank-three function edits have
relative MSE around `.0004`–`.0011`. A gauge fitted without the bit-flip
edges assigns roughly `99.95%` of the A/B/C finite-effect attribution to
the matched component and predicts the full edit better than SVD axes.
Training-frequency reduction favors common-query edit fidelity while
uniform weighting better preserves the rare whole-combination slice.
These are input interventions and surrogate-coordinate attributions,
not physical head or neuron patches.
The
[branch-intervention control](SHARED_PAIR_BRANCH_CAUSAL_RESULTS.md) shows
why: the task-faithful rank-three space predicts constant-one QK1/QK2
edits poorly in both token codings. Joint fitting to ordinary and edited
tables needs more key directions, and exact gauge rotations cannot
recover directions omitted by the ordinary reduction. The
[546M full-context QK2-response comparison](FULL_MODEL_BRANCH_RESPONSE_RESULTS.md)
tests four matched rank-256 QK1-key bases on 24 disjoint documents per
corpus. Branch-aware fitting yields a small FineWeb gain over ordinary
function-aware fitting, with no established external-corpus advantage;
both beat uniform key-table SVD on all three edited-logit response panels.
The [held-out QK2 edit-family check](FULL_MODEL_HELDOUT_BRANCH_EDIT_RESULTS.md)
shifts QK2 scores across neighboring causal keys before the QK1·QK2
product. It preserves the small FineWeb branch-aware gain but finds no
external-corpus advantage over ordinary function-aware fitting. Both
function-aware bases substantially outperform uniform weight-only SVD in
full-model edited-response prediction. This is compression/response
fidelity, not yet a legible causal decomposition.
The [full-context sensitivity gauge](FULL_CONTEXT_SENSITIVITY_GAUGE_RESULTS.md)
keeps the rank-256 projector fixed and rotates its axes using gradients of
next-token log probability through all 18 layers. It substantially reduces
held-out effective sensitivity coordinates on FineWeb, WikiText-103, and
TinyStories, but only about half the axes match closely across independent
starts. Coarse key-token-form enrichment is mixed and raw energy scores
are tail-dominated, so this remains a sparse downstream coordinate prior,
not a canonical or selectively causal mechanism.
The [finite full-context-axis tests](FULL_CONTEXT_FINITE_AXIS_RESULTS.md)
select two stable punctuation-associated axes from FineWeb training gradients.
Natural-key deletions are position-confounded; a fresh same-position
punctuation-versus-word substitution shows strong full-vocabulary
punctuation selectivity for one axis across three corpora, but its
next-token-target effect is less consistent. An older pair-response axis is
at least as selective after a partial training-energy-matched deletion.
This establishes a narrow finite association in the reduced model, not
that the full-context sparsity prior recovers a superior or unique mechanism.
The [unseen-key-ID red team](FULL_CONTEXT_UNSEEN_TOKEN_RESULTS.md) uses eight
frequency-matched punctuation/word pairs absent from the gauge-fitting key
positions. Both selected axes retain finite punctuation-associated output
effects on every pair, with good native-to-reduced input-edit fidelity.
The older pair-response axis is still at least as selective; sparse
full-context gradients add no demonstrated legibility gain for this label.
The [frozen output-signature test](OUTPUT_SIGNATURE_TRANSFER_RESULTS.md)
uses four unseen-token pairs on FineWeb validation documents to propose a
signed logit direction and output-category contrast, then tests the other
four pairs on three corpora. Axis 7 has only weak same-corpus direction
alignment and no established external-corpus output rule; the pair-response
axis is also incoherent. The punctuation effect is real as a conditional
magnitude in the reduced model, but it should not be named a
punctuation-prediction circuit.
A [diverse-task attention benchmark](DIVERSE_TASK_CANONICAL_RESULTS.md)
now trains five eleven-output toy models and withholds five complete output
tasks from reduction fitting. Rank-two cuts cannot preserve all three designed
count variables; response fitting improves ordinary output error but fails on
the rare composition. At a fixed rank-three reduced function, key L1 and
fit-task sensitivity L1 both recover the known count directions across five
seeds, with no meaningful extra held-task gain from the sensitivity prior.
The [dense-code continuation](DIVERSE_TASK_CANONICAL_RESULTS.md#dense-token-red-team)
is now complete across five new trained seeds. Key-entry L1 fails to recover
the three designed variables under overlapping token attributes, whereas a
response-sensitivity L1 gauge fitted only to three XOR outputs recovers them
and produces sparser sensitivities on eight unseen outputs. All gauges leave
the fixed rank-three reduced function unchanged. This supports supervised
task-response geometry as a prior. Fresh finite-axis edits on eight held-out
outputs preserve its sparsity advantage at .05 and .25 of a matched axis dose,
but lose it at full deletion. The gradient prior is local; physical head
identity and 546M transfer remain untested.
The [finite-fit and token-edit continuation](DIVERSE_TASK_CANONICAL_RESULTS.md#unseen-token-replacements)
shows why the edit distribution matters. Fitting finite coordinate-deletion
sparsity or activation-weighted gradients improves those deletion objectives
but no longer recovers the known variables. On a separate discrete-token family
of 24 directed one-attribute token replacements, the unweighted XOR-response
gauge instead predicts eight held-out output changes using one axis with
.00008 mean normalized error, versus .166 for SVD and .440 for key L1;
all 120 seed–pair cases are below 1% error. The full rank-three cut has
.00030 native edit-response error. This is strong toy-level evidence for a
task-supervised functional variable, not physical head identity or 546M
transfer.
A first [balanced 546M output-stratum test](BALANCED_OUTPUT_GAUGE_RESULTS.md)
holds two next-token-form categories out of fitting while rotating the same
fixed rank-256 QK1-key projector. Full-context gradient sparsity transfers
to those strata on FineWeb, WikiText-103, and TinyStories, but train-selected
category axes do not preserve their labels on disjoint documents. An
all-category oracle has similar sparsity with substantially different axes.
This is generic gradient compression, not a validated semantic component;
the coarse-category detour stops before finite axis selection.
An [explicit-task 546M comparison](EXPLICIT_TASK_GAUGE_RESULTS.md) screens
seven prompt families and retains five with 100% forced-choice accuracy on
independent prompts. The fixed rank-256 QK1-key cut preserves those choices.
A gauge fitted on agreement, number continuation, and bracket closure makes
their gradients sparse but transfers poorly to whole-family antonym and
capital holdouts; adding each family chiefly improves itself. Training-selected
axes retain task-label gradient enrichment and weak finite-dose selectivity,
but the capital and antonym axes peak at the template's first token “The.”
On 23 successful minimal answer-changing token swaps, full QK1-key path
patches alter the margin by only .013–.028 on average versus native input-edit
changes of 5–18; selected-axis patches are smaller still. Even full layer-0
attention-output patches are small for most families. Stop optimizing these
first-layer axes for those tasks: the causal mediation gate fails despite
supervised sparsity gains.
The
[all-head real-model comparison](ALL_HEAD_SHARED_KEY_RESULTS.md) shows useful
shared-key compression; the [function-aware fit](FUNCTION_AWARE_SHARED_KEY_RESULTS.md)
improves natural fidelity but loses much of its advantage under a QK2 edit.
The [exact-gauge prior comparison](ALL_HEAD_GAUGE_PRIOR_RESULTS.md) and
[label transfer test](ALL_HEAD_LABEL_TRANSFER_RESULTS.md) show that sparse
coordinates and label enrichment can disagree. The
[full-model causal test](ALL_HEAD_COUNTERFACTUAL_CAUSALITY_RESULTS.md) finds a
small, reproducible punctuation-edit effect for one selected component;
whether its proposed query/key direction generalizes is a separate test.

The toy work has reached its immediate stopping criterion: a function-level
canonical graph survives a dense-token red team and task-faithful pruning,
while [small QK-branch doses](SHARED_PAIR_BRANCH_CAUSAL_RESULTS.md) expose
causal directions outside that semantic subspace. The matched-rank 546M
QK2-response test is now complete for two edit families, one held out from
branch-aware fitting. The first full-context gradient gauge and matched
finite-axis tests are also complete. The gradient prior passes a narrow
punctuation response test but does not beat the prior pair-response gauge.
The held-out key-ID and named-output-signature tests are complete. The
punctuation-axis detour has reached its stopping point: neither gauge
produced a stable broad output rule. The first diverse explicit-task toy
comparison is complete: eleven competent outputs, five whole-task reduction
holdouts, dense-token red teaming, and exact rank-three gauge priors. Its
next discriminator is an attention-dependent task family whose answer
changes are materially mediated by the proposed attention path, or a
later-layer canonical target. The coarse 546M output-stratum panel ruled
out naming axes from gradient sparsity alone; the explicit-task panel adds
that competent task labels and transferable gradient selectivity still do
not establish causal use of first-layer QK1 keys. Continue to compare
weight-only, data-aware, and causal priors at matched rank/fidelity only
after such a path-specific behavior gate passes.
The [later-layer attention-locus scan](LATER_LAYER_ATTENTION_LOCUS_RESULTS.md)
now passes a **path-sufficiency** gate on these competent tasks: full
attention-output patches at layers 11 or 13 have large margin effects, and
frozen single-head selections survive 21 new token-swap pairs. The effect
is almost entirely at the final prompt position. This redirects the next
matched canonicalization comparison to later-layer 128-dimensional head
outputs, with native-model finite interventions and whole-task holdouts.
Head-output sufficiency does not yet prove necessity, source computation, or
a human-readable component; these remain separate validation gates.
The first [later-head matched-rank metric comparison](LATER_HEAD_REDUCTION_METRIC_RESULTS.md)
uses weight-only, natural-corpus, task-edit, mixed, and random rank-4/8/16
orthogonal cuts on those 128-dimensional vectors. Weight or edit-informed
rank-4 bases preserve the held-out bracket and agreement head-path effects;
corpus PCA misses most of them. Capital and antonym remain poorly preserved
even at rank 16. Forced-choice accuracy can stay perfect while a selected
head path is lost, so subsequent legibility work must retain finite
path-effect fidelity as a separate gate. The bracket and agreement heads are
the immediate fixed-function gauge testbeds; their exact within-subspace
rotations can compare sparsity objectives without changing the rank cut.
The [exact later-head gauge comparison](LATER_HEAD_EXACT_GAUGE_RESULTS.md)
now executes that test: exact rank-4 rotations of bracket and agreement
head responses compare output-weight L1, edit-vector L1, local-effect
concentration, SVD, and 64 random orientations. A single intervention-
supervised axis reproduces the full rank-4 finite path within about 0.4–2%
on new token edits and within 0.4–0.8% on 26 new prompt formats. Weight L1
does not provide the same coordinate. Whole-target-family exclusion fails:
neither rotation nor axis selection from unrelated tasks recovers the target
path. Thus this is a validated *supervised* gauge for two selected heads,
not unsupervised task discovery or a canonical decomposition of the entire
attention function. Reverse head-state patches now confirm the selected
axis has the expected opposite-sign effect on 38/38 fresh pairs, but even
restoring the full native head state flips the edited answer back in only
3/38 cases. The axis is a compact influential path, not a demonstrated
necessary answer circuit. Broader natural-language tasks and redundant-
path analysis are the next causal and generalization gates.
The [unlabeled natural-activation red team](LATER_HEAD_UNLABELED_GAUGE_RESULTS.md)
tested PCA, activation L1, fourth-moment rotations, 8,192 short FineWeb
contexts, independent halves, external corpora, and balanced multicorpus
fits inside the same exact rank-4 subspaces. A 512-context bracket success
was destroyed by removing one high-norm natural sample; the 8,192-context
fit and both large halves failed finite path tests. Balanced multicorpus
fits did not recover bracket and gave only variable partial agreement
recovery. This specific unlabeled-activation route lacks a stable axis.
The next legibility test should use diverse **unlabeled interventions**
(token substitutions and context edits) with whole-edit-type and whole-task
holdouts, rather than more generic activation sparsity. It must retain
finite head-path fidelity and show sample/seed stability before claiming
new-task discovery.
The [answer-free intervention follow-up](LATER_HEAD_UNLABELED_INTERVENTION_RESULTS.md)
tests exactly this next step on two independently generated, disjoint
800-pair panels spanning ten edit types. At the same fixed rank-4 head
cuts, raw edit-response PCA and edit L1 recover bracket and agreement
axes from input edits without answer labels; independently fitted axes
nearly coincide and one-axis finite patches reproduce the rank-4 path
within about 1.5–3% on new prompts. Removing the matching edit type
substantially degrades path fidelity. The top-decile edit-response cluster
also gives input-side candidate labels, but full-head response norm is
nearly as informative and the heads were preselected with task knowledge.
This establishes a useful answer-free *intervention prior* under covered
edit types, not genuine discovery of a novel task. A rare-edit prevalence
sweep shows that a small explicit coverage floor can be decisive; increasing
sample count alone cannot recover an edit type assigned zero mass.
A genuinely blind next pass should scan many heads/layers, compare with
head-norm controls, and reserve task-family labels for downstream validation.
The [first all-head answer-free scan](ALL_HEAD_UNLABELED_EDIT_SCAN_RESULTS.md)
now covers 36 native heads in four later layers with independent 800-edit
panels. Its top unlabeled-energy head is the known bracket locus; other
top heads show reproducible punctuation, bracket, and noun-number response
clusters. Axis clustering sometimes improves over full-head norm and
sometimes does not. These are input-side candidate variables only; the
next gate is frozen output predictions and finite patches on disjoint
contexts before claiming a newly found behavioral mechanism.
The [secondary-head finite-path gate](SECONDARY_HEAD_FINITE_RESULTS.md)
patches layer-11 bracket candidates 5 and 6 on independent validation
edits. Their full-head directional logit paths are about 4.2% and 2.0% of
the native bracket edit, jointly 6.2% with negligible interaction; the
known layer-13 head 8 carries about 31%. Unlabeled edit-response energy
therefore finds additional input-sensitive heads but does not by itself
find a comparably strong output mechanism. The same edit families were
used for axis fitting and validation, so new token identities and frozen
output predictions remain required.
The [unseen token-ID edit test](UNSEEN_TOKEN_EDIT_TRANSFER_RESULTS.md)
shows why pair identity matters in addition to context: both 800-edit
panels reused one parenthesis/square substitution, despite disjoint
full prompts. A frozen bracket axis loses about 45% of the rank-four
head path on two new brace-swap identities, with a much larger failure
on brace→parenthesis; the rank-four cut itself remains faithful. The
model is not competent on brace-closing prompts, so this is a path
generalization limit rather than a competent task result. The agreement
axis transfers within about 3% of rank four on four unseen noun-number
swaps for which the model is competent. Interpret stable axes against
both context and token-identity holdouts.
The [competent bracket-identity follow-up](COMPETENT_BRACKET_IDENTITY_TRANSFER_RESULTS.md)
now finds six new token-ID substitutions that preserve the parenthesis/
square relation and pass independent native forced-choice validation.
The frozen bracket axis retains the rank-four head path within 4.4%
pooled over 21 new-prefix/new-expression pairs; rank four itself is
within 2.3% of the native head. Eleven pairs have both expected closers
as top-1 next tokens, and all expected closers rank top three. This
shows a real tokenization-identity generalization within one relation,
while the brace failure remains a limit on broader bracket transfer.
The [bracket edit-span follow-up](BRACKET_EDIT_SPAN_RESULTS.md) adds
brace edits to the fitting measure without using output labels and
finds a second strong direction inside the same exact rank-four cut.
On held-out brace contexts and old-pair new formats, a rank-two
response group retains the rank-four finite head path to about 0.1%
and 0.7% relative error respectively, while every rank-one choice
misses a swap type. This supports grouped coordinates as a
representation of broader input-side response geometry, but the model
is not competent on the brace-closing task, so semantic and behavioral
claims require a different validation panel. In a
[further frozen identity-holdout test](BRACE_COMPETENCE_AND_GROUP_TRANSFER_RESULTS.md),
the rank-two group retains the rank-four path within 0.6% on 21 native-competent
parenthesis/square prompts using six new token-ID substitutions, versus 4.4%
for the old one-axis gauge. A matched rank-two control fitted on the original
parenthesis/square edits alone does equally well (0.56%) and captures held-out
brace paths within .19–.31%. The second direction was already present at
.73% of original edit energy; **retaining rank**, rather than adding brace
examples, explains most of the gain. Weight-SVD rank two misses 26% on the
competent holdout and random planes miss 55% median. A 19-pair code-like brace
screen finds zero native brace forced-choice successes, making this a
transferable response group, not an established brace-rule mechanism.
A [reduction × group factorial](BRACKET_REDUCTION_GROUP_FACTORIAL_RESULTS.md)
crosses five prefit rank-four layer-13/head-8 cuts with the same unlabeled
80-edit covariance group, then tests 21 competent unseen token-ID prompts.
Weight-only rank four misses 2.29% of the native finite head path and its
two-axis group misses 2.32%; corpus PCA rank four misses 80.25% even though
its own two-axis group reproduces that reduced path within 1.25%. All five
cuts retain 21/21 forced-choice answers. This demonstrates why **stage-one
causal fidelity must gate stage-two legibility**: an exact rotation or
group inside a bad cut cannot restore omitted head-state information.
The independent [agreement-head factorial](AGREEMENT_REDUCTION_GROUP_FACTORIAL_RESULTS.md)
repeats the gate at layer 11/head 3 on 20 competent lexical and format-shift
pairs. It fits an answer-free two-axis group from 44 noun-number edits after
excluding every validation token-swap identity. Weight, task-edit, and mixed
rank-four cuts miss 4–5% of the native head path, whereas natural-corpus PCA
misses 72%. Corpus's two-axis group nevertheless misses its *own* rank-four
path by only 0.54%. Corpus loses two forced-choice pairs; a random cut
misses 96% of the head path yet retains all 20 answers. The reduction gate
is therefore a cross-task mechanistic issue, not a bracket-only artifact.
The [larger natural-text rank-four KL panel](LATER_HEAD_RANK4_KL_SCALE_RESULTS.md)
evaluates the five frozen cuts on 576 paired held-out windows per selected
head across FineWeb, WikiText, and TinyStories at lengths 32 and 64. The
weight cut also has lower equal-panel mean KL for the bracket head
(0.000977 vs corpus PCA's 0.002467), while agreement's aggregate ranking
is unresolved (0.001839 vs 0.001763; paired interval crosses zero).
Two apparent corpus wins at individual corpus/length cells are uncertain.
This broadens the fidelity gate beyond hand-selected edit paths without
proving that the small 256-context unweighted PCA fit is the best possible
dataset-aware reduction. The next useful comparison should improve the
dataset-aware *objective or fit coverage*, rather than merely enlarge its
evaluation panel.
The [output-aware rank-four follow-up](OUTPUT_AWARE_HEAD_REDUCTION_RESULTS.md)
does both: fit 2,048 unlabeled FineWeb head states at two lengths, optimize
the orthogonal projector for `E||W(I-P)z||²`, then evaluate on the same
576 held-out windows per head and competent edit paths. The bracket
cut improves equal-panel next-token KL from 0.000977 (weight-only) to
0.000803, with a paired 95% interval for the difference entirely below
zero, but loses 66% of the native bracket head path versus 2.3% for
weight-only. Agreement's KL gain is unresolved and path loss rises from
5.2% to 8.9%. This is the clearest observed reduction-metric conflict:
dataset-aware output fidelity and targeted mechanistic fidelity cannot
be conflated. The next discriminator is a predeclared mixture of natural
output and edit-response moments with both KL and finite-path gates.
The [joint moment frontier](JOINT_HEAD_MOMENT_FRONTIER_RESULTS.md) now
executes that discriminator at rank four. A predeclared seven-point
natural-output/edit-response mixture yields a bracket λ=.1 cut with
equal-panel next-token KL 0.000810, below weight-only's 0.000977,
while matching the native head edit path within 2.05% versus weight's
2.29%; paired KL interval is entirely favorable, while the small path
difference is uncertain. A new 24-expression, six-token-identity
confirmation retains 2.10% path error versus 2.83% for weight and
64.57% for natural-only. Agreement does not show the same improvement:
the best mixed edit path is 8.05% versus weight-only's 5.16%, and its
KL ranking remains unresolved. The useful compromise is therefore
head/task-specific. The next stage is to compare sparse, grouped, and
intervention legibility priors *within the faithful joint bracket cut*,
keeping the cut and its KL/path gates fixed.
The [fixed-cut bracket legibility comparison](JOINT_BRACKET_LEGIBILITY_RESULTS.md)
does so at the joint λ=.1 rank-four reduction. Answer-free PCA,
normalized PCA, edit L1, and varimax of 80 bracket-edit responses
recover almost identical top axes; train/validation raw-PCA axis cosine
is 0.99997. Two PCA axes reproduce the joint cut's finite path on 24
fresh competent expressions within 0.78%, versus 5.84% for one axis,
19.51% for the original two coordinates, and median 15.34% across 16
matched random rotations. Exact basis rotations leave the projector
unchanged to `3.1e-15`; finite full-vocabulary logits agree within
`6.7e-6`. The axis detects bracket edits in an independent ten-type
validation mixture, but the full-head norm already detects them
perfectly. This is compact path representation, not genuine new-task
discovery or a unique canonical gauge.
The [nested-bracket transfer and removal test](NESTED_BRACKET_TRANSFER_RESULTS.md)
challenges that same frozen group with an unseen *inner versus outer
closer* relation. Native forced-choice competence is 12/12 inner and
10/12 outer, though expected closers are seldom top-1. The joint cut
misses native finite head paths by 2.8% and 6.0%; its two-axis
answer-free group reproduces its own cut within 1.7% and 1.3%.
Its one-axis version misses 77.7% of the inner path. Removing the
two-axis group leaves 4/12 outer pairs correct, like removing the
whole cut, while removing the complementary axes leaves 10/12.
Two of 16 random rank-two groups are comparably disruptive, so this
is a localized causal contribution, not unique necessity. **Stop the
single-head bracket detour here**; the next experiment should use
task-blind head/response selection and withheld behavior families,
or known-rule multi-task toy language, rather than further tuning
bracket prompts.
The [trained-toy rank-selection control](THREE_COUNT_RANK_SELECTION_RESULTS.md)
fits uniform weight-only, usage-weighted, and mixed token functions at ranks
one through six on ten independently trained three-variable attention models.
An answer-free within-pair edit-response criterion selects rank three under
all three measures and three thresholds; every selected cut retains the
withheld rare combination and all four task outputs. Rank two has large edit
error and often fails the rare task. This checks the decision rule in an easy
regime, but does not yet test whether it retains a *low-energy* competent
rule when high-energy nuisance responses compete for rank.
An [exact internal-bond gauge stress test](THREE_COUNT_BOND_GAUGE_STRESS_RESULTS.md)
then rescales a singular coordinate in a factorization of those same ten
trained functions while compensating in the other factor. The complete
function and task are mathematically unchanged. At a 5% error budget,
local factor-energy truncation selects rank two in every seed under
`ε=.1` or `.03`, with only 10% rare held-out accuracy; the full-function
SVD selects rank three and retains 100%. Answer-free decoded edit fidelity
also retains behavior but can require rank four through six in the bad
gauge. This demonstrates the need to fix a functional metric and gauge
*before* sparsity/group optimization. It is an exact factorization
control, not evidence that the trained physical heads use that gauge.
A [trained QK-quotient gauge test](DUAL_QK_QUOTIENT_GAUGE_STRESS_RESULTS.md)
checks the analogous choice at an actual attention branch. Ten trained
necessary-two-head models supply exact rank-two QK1 score quotients;
six invertible quotient gauges per head leave each uncut function unchanged.
At rank one, local key-factor SVD plus output-aware amplitude refitting can
range from 1.5% to 45% mean ordinary head-output error by gauge, whereas
environment WTG stays at 1.0% and its reduced output changes by at most
`1.2e-12` across gauges. One rare-rule seed still fails under WTG's tight
rank-one budget. This verifies gauge-stable reduction on trained attention
without conflating quotient coordinates with the physical width-four bond
or claiming semantic head identification.
A [full-model branch-dose test](ALL_HEAD_BRANCH_DOSE_FULL_MODEL_RESULTS.md)
now propagates four saved rank-256 shared-key cuts through all 18 layers
while interpolating every first-layer QK2 score toward one. On 32 exactly
paired held-out windows per corpus and strength, a 50:50 natural/branch
pair-response fit slightly improves KL to the branch-edited reference on
FineWeb across strengths and on TinyStories at quarter strength; WikiText
is mixed. The branch edit itself raises native CE sharply, and no
consistent CE gain follows from the improved fidelity. This separates
intervention-preserving reduction from both task improvement and
legible causal components.
The [finite punctuation-head red team](PUNCTUATION_HEAD_FINITE_RESULTS.md)
finds only a small output path: complete head-13/3 and 13/5 patches follow
roughly 5% and 4% of the punctuation edit's full-vocabulary logit change,
and their leading axes follow about 3% and 2%. In contrast, the known
bracket head 13/8 follows about 31% of bracket-edit logit change and its
one axis retains essentially all of that path. The punctuation clusters
are selective input responses, not yet strong newly discovered circuits.
This failure mode should be built into known-rule toys when comparing
response sparsity with intervention and downstream priors.
The [joint native patch](LAYER13_JOINT_PATCH_RESULTS.md) checks whether
the two punctuation heads cancel: together they follow about 9% of the
input edit's logit change, nearly the sum of their separate paths; all
nine layer-13 heads follow about 12%. The weak path is not explained by
mutual cancellation of heads 3 and 5 at this intervention dose. Stop
single-head punctuation interpretation here unless a stronger frozen
output prediction appears; use it as a negative control in the toy prior
comparison.
A concrete position-to-position path hypothesis could separately
justify tracing earlier-position QK1-key effects. A
separate branch-level experiment can define causal response signatures
across multiple edit families and corpora, including QK1 or two-branch
perturbations, and optimize
condition-wise legibility only after reduction fidelity is gated on
disjoint corpora and downstream logits. The earlier query/key-form
factorial already rejected pair-level directional labels, so
selected-pair and full-model readouts must remain separate.

The [36-head finite path and coverage screen](TASKBLIND_HEAD_PATH_SCAN_RESULTS.md)
tests the next selector without answer labels. The downstream mean-path head
ranking is stable across same-type panels, but it is dominated by broad
responders and places the known bracket specialist fourth. A greedy
three-head maximum-coverage rule selects that specialist and improves
held-out-example output-path coverage over mean-path and input-energy
rankings. When the entire bracket edit family is excluded from fitting,
the specialist drops out of the top five. This is an explicit stopping
criterion for more scalar all-head ranking on the present ten types:
progress now requires a withheld-task toy or a new behavior prediction
predeclared from unlabeled signatures, and the reduction-metric × legibility
comparison should be judged on that gate.
The [ten-model unseen-query toy test](UNSEEN_QUERY_COVERAGE_RESULTS.md)
gives the reduction-first rule a sharp support boundary: two rank-two
functions fit every observed ordinary-query cell exactly but differ from
10% to 100% accuracy on the withheld swapped-query combination. Weight-only
information chooses a successful completion; a zero-support data metric
cannot distinguish them. An exact sparse gauge aligns the known A/B key
variables under either completion and cannot restore the missing query
mapping. Next use finite unlabeled corpus coverage and a withheld output
rule, rather than treating exact expected positive mass as finite-data
evidence.
The [finite usage-measure sweep](FINITE_USAGE_METRIC_COVERAGE_RESULTS.md)
uses ten trained three-bit-plus-parity models and five independent sampled
unlabeled corpora at six sizes. At rank two, empirical usage improves rare
held-out accuracy as coverage grows but plateaus near 53%; a half-uniform
mixture lowers broad table error while cutting that rare accuracy to 19%.
At rank three, all 900 weighted fits are at least 99.64% rare-accurate,
and an answer-free all-pair edit criterion selects rank three in all
1,800 threshold decisions. The exact weight-derived function and full-table
SVD start remain available, so this is finite *measure* estimation, not
data-only function discovery. Stop using this easy rank-three toy as a
discovery claim; construct a weak competent rare rule or strong nuisance
and withhold its edit family from selection.
The [four-rule weak-output control](WEAK_RULE_CANONICALIZATION_RESULTS.md)
trains ten new four-head count-attention models, then scales one competent
output by 0.1 without changing its decisions. At rank three, uniform,
usage, and mixed cuts report under 0.5% global edit error but reduce the
weak bit to chance in every seed. A per-output relative fidelity gate
chooses rank four for all four reduction metrics and retains every task;
an output-balanced rank-three fit preserves the weak bit only by losing
another rule. Exact sparse rotations of those faithful rank-four cuts
recover all four known token axes across seeds, including when D-token
edits are excluded from the rotation objective. Because the full weight
table enters reduction and SVD already isolates D, this is a metric and
coordinate control, not unseen-rule discovery. Stop treating global edit
error as sufficient; the next test needs dense mixed token coding, a
weak competent rule, and a genuinely withheld edit family.
The [dense-coded weak-rule follow-up](DENSE_WEAK_SHARED_ATOM_RESULTS.md)
trains ten new four-head models on nonorthogonal four-bit token features.
After scaling a competent D output, uniform, usage, and mixed rank-three
cuts pass a 1% global all-pair edit gate in every seed while preserving
only 64.5% D prediction agreement. A per-output gate retains rank four.
At fixed rank four, label-free edit-sparse rotations recover the designed
overlapping four-atom function graph with .976 graph-cell accuracy, versus
.719 for SVD and .601 for random rotations; a prior withholding direct
raw-D flips reaches .968. The full weight-derived function still enters
the cut and the same-D-bit edits retain mixed D information, so this is
function-level graph recovery, not physical-node or wholly unseen-rule
discovery. The next gate is a shared-dictionary/overlapping-group model
with a whole-rule holdout, compared at matched error and rank.
The [exact shared-dictionary identifiability test](DENSE_WEAK_GROUP_IDENTIFIABILITY_RESULTS.md)
shows why group sparsity alone is not that gate. Across 40 fixed rank-four
cuts, an output-row dictionary and an oracle raw-bit dictionary represent
the identical function to below `3.2e-13` maximum error. The former is
nonorthogonal, well conditioned, and needs only eight active output-to-atom
links; the intended reusable-bit graph needs eighteen. Fitted edit-sparse
gauges nearly recover the latter and have far higher single-bit edit
purity. Thus raw sparse membership favors output-specific atoms, while
intervention structure favors reusable input-rule atoms. A learned group
model must include an explicit input/intervention prior and earn its place
on withheld compositions; static support minimization is insufficient.
The [internal-head no-readout test](UNLABELED_HEAD_RULE_RECOVERY_RESULTS.md)
fits rank-four dictionaries to QK1×QK2×V head responses without consulting
the D output readout or task labels. Weight-only and unlabeled usage fits
both preserve >99.98% native D decisions. At fixed function, all-pair
edit-sparse gauges reach .994 raw-bit atom alignment and ~.003 D-flip
component-response error, versus ~.691 and ~.323 for SVD. Excluding the
eight direct D-flip edits from gauge fitting degrades the latter to .157
weight-only and .124 usage-weighted, even with 32 starts. The model was
trained on D and other training edits still contain mixed D information:
this is an internal representation audit, not wholly unseen-task discovery.
Next use independent rule compositions and whole-composition holdouts to
test whether a learned group architecture adds value over exact rotations.
The [few-shot new-task transfer check](NEW_TASK_FEW_SHOT_TRANSFER_RESULTS.md)
tests four raw-bit majority rules and six XOR-of-majorities compositions
that the trained models never output. With 64 labels per new task to choose
one atom or one atom pair, all-pair edit-sparse head dictionaries reach
~99.7% primitive and ~99.4% XOR accuracy on 20,000 new sequences per
model, compared with ~74.6% and ~65.1% for SVD. All bases span the same
~99.995%-complete raw-bit subspace; this is sparse-probe accessibility,
not extra information or automatic task invention. Withholding direct D
flips from gauge fitting worsens D and composition transfer. The next
gate needs an actual learned group featurizer and a whole-composition
holdout, with task prediction and intervention fidelity kept separate.
The [learned static overlapping-group comparison](LEARNED_OVERLAPPING_GROUP_RESULTS.md)
fits nonorthogonal four-atom dictionaries at exactly fixed function fidelity.
On eight model seeds held out from ratio selection, group-link L1 alone
chooses the minimal eight output-specific links and gives .939/.886
one-atom/XOR few-shot transfer. A preselected edit weight of 80 plus 32
starts recovers the intended 18-link graph in seven of eight weight-only
and eight of eight usage-weighted fits; a label-free edit-gauge warm start
recovers it in all seeds. The simple orthogonal edit gauge was already
~99.99% accurate on these new tasks, so the learned static group model adds
little measurable transfer here. Stop scalar penalty tuning on this toy;
test sample-dependent gates on an independently held-out composition family
or move the prior to the flagship model with downstream causal validation.
The [36-head intervention-covariance atlas](ALL_HEAD_JOINT_EDIT_GEOMETRY_RESULTS.md)
tests an intervention-defined canonical prior across four later layers of
the 546M model. Joint diagonalization reduces held-out cross-axis edit
covariance relative to response PCA on 36/36 equal-type and 34/36 raw-energy
fits, with median independent-panel axis cosines .988/.982. Yet the
train-PCA rank-four cuts retain only .380 mean validation response energy;
just four heads exceed .75 coverage. A whole-edit-type exclusion on those
four shows specialist dependence: bracket head 13/8 concentrates 99.6%
of its bracket response on one equal-type axis with bracket examples,
but only 65.8% with that edit family omitted. The faithful heads are
already known bracket or weak punctuation loci. Do not claim a new
circuit from this input-side geometry; improve stage-one fidelity or
freeze and test a genuinely new downstream prediction.
The [all-head reduction-fidelity frontier](ALL_HEAD_RANK_FIDELITY_FRONTIER_RESULTS.md)
corrects the interpretation of head-state coverage by applying each
head's learned `c_proj` block. On the known bracket head, rank-four
pooled-edit PCA loses 6.4% head-state but only 2.74% aggregate immediate
output response. Yet it loses 93–97% of eight other edit types separately.
Across 36 heads, weight-only rank four has .813 mean aggregate output
relative MSE, pooled edit PCA .592, and equal-type edit PCA .642; mean
worst-type errors remain .936/.922/.904. In the tested spectral family,
a fixed 50:50 weight/equal-output-type measure needs median rank 96 for
all ten types to pass 10% immediate-output error. A gauge cannot restore
families excluded at reduction; decide which counterfactuals matter,
then gate them on disjoint inputs and final logits.
The [path-selected reduction follow-up](ALL_HEAD_PATH_SELECTED_REDUCTION_RESULTS.md)
uses native full-model path magnitude to select three edit families per
head without answer labels. Across 36 heads, rank-32 cuts lower mean
selected-family immediate-output error from .204 for pooled edits to .159,
while raising aggregate error from .223 to .269 and worst unselected-family
error from .532 to .589. A finite full-model test on L9H7 reproduces the
selected-family path-fidelity benefit but triples bracket-path error
(.122 to .372). Path relevance is therefore a useful *choice of metric*,
not evidence that the resulting reduced coordinates are legible or that
all behavior is covered. The next gate must preserve ordinary forward
logits and fresh edit families before choosing a sparse gauge.
The [training-only floor rule](PATH_FLOOR_GUARDRAIL_RESULTS.md) makes the
coverage choice adaptive without validation tuning. At rank 32 it improves
selected-family local output error from .204 to .171 across 36 heads while
mean worst unselected error stays .534 versus .532 pooled; seven individual
heads nonetheless exceed the training rule's .02 validation margin. For
L9H7, its chosen .5 floor cuts bracket-path error from .372 at a fixed .1
floor to .193, still above the .122 pooled baseline. This is a partial
stage-one fidelity improvement, not evidence for a legibility prior.
The [whole-edit-family transfer test](NEW_FAMILY_PATH_TRANSFER_RESULTS.md)
introduces modal, connective, tense, and code-comparison substitutions
absent from both fitting and floor selection. At rank 32, L9H7's
training-floor cut has .437 mean finite final-logit path MSE on these new
families versus .428 pooled; at rank 64 the two are .183/.184, and at
rank 96 .069/.068. The floor's familiar-family improvement does not
generalize to these families. Higher rank helps markedly, but rank 96
retains three quarters of the 128-dimensional head. Proceed to stage-two
legibility optimization only under an explicitly declared fidelity gate,
not by assuming that familiar-edit local coverage transfers.
The [finite downstream path gate](FINITE_PATH_GATE_RESULTS.md) replaces the
local constraint for L9H7 with an absolute 10%-per-family final-logit
head-patch error and mean base-prompt KL below .001, using only training
edits. Among 20 spectral candidates it selects pooled-edit rank 96;
no rank-32 or rank-64 candidate passes. The frozen cut passes all ten
familiar families on independent validation (worst mean error .067)
and eight of ten wholly new designed families across two panels; modal
and yes/no polarity miss the .10 gate narrowly (.104 and .101). This
partly repairs transfer by retaining 75% of the head's 128 dimensions,
not by revealing a small canonical computation. Stage two should compare
exact rotations within this fixed faithful subspace, while stage one
separately seeks a lower-rank projector that passes the same gates.
The [rank-96 exact-gauge prior comparison](RANK96_EXACT_GAUGE_PRIORS_RESULTS.md)
does that controlled stage-two test. Output-isometric rotations preserve
the fixed reduced `c_proj` response to about 1e-15 relative error.
Weight varimax makes decoder entries modestly sparser, while edit
activation varimax sharply concentrates familiar interventions and
separates their type-energy profiles. Yet pooled PCA concentrates the
ten unseen-family edit responses more, and refitted varimax axes have
only .443 mean absolute cosine agreement with independent-panel axes,
close to PCA's .422. Exact gauge sparsity and type selectivity are
therefore not enough to name stable reusable components.
The [finite top-k axis-pruning red team](RANK96_CAUSAL_AXIS_PRUNING_RESULTS.md)
uses those same exact gauges to patch only the strongest 8, 16, or 32
output-isometric axes of independent edit responses into the full model.
At 16 axes, pooled-edit PCA beats edit varimax in final-logit path
error on familiar edits (.211 vs .241) and ten unseen-family edits
(.285 vs .316), with nine of ten unseen family means favoring PCA.
The earlier inverse-Simpson sparsity and type-purity gains do not
translate into better causal top-k compression. Stop treating this
varimax prior as evidence of interpretable reusable components; the
next meaningful comparison is group-level, answer-level, and
matched-dose rather than another single-axis sparsity score.

The [paired construction-and-dose follow-up](PREROPE_AS_CONSTRUCTION_DOSE_RESULTS.md) tested the previously discovered first-layer QK1 `" as"` response group on 24 fresh sentences with two occurrences of tokenizer ID 355. Editing first versus second `" as"` had different effects on the same suffix at matched post-`c_proj` residual energy, but three Haar rank-ten controls showed the same pattern and comparable KL. Context-dependent lexical activation survives; a distinct grammatical causal channel does not. This meets the stopping rule for this single-lexeme branch. Return to multiple learned rules, whole-task holdouts, and matched-dose group interventions.

The [query-conditioned multi-composition benchmark](CONTEXTUAL_COMPOSITION_CANONICALIZATION_RESULTS.md) advances that broader loop. Five newly trained eight-head bilinear models solve four primitive-majority, six XOR, and six AND tasks essentially perfectly. Holding the entire XOR query family out of featurizer fitting, rank-six weight-only and sequence-aware cuts preserve roughly 99% seen-task accuracy but only 82.3%/82.0% XOR accuracy; including XOR weights in an explicitly leaky oracle cut yields 97.4%. More examples of the seen tasks cannot substitute for coverage of a missing task family. Within the exact full eight-dimensional function, top-four sample-dependent gates on physical heads retain 97.6% XOR accuracy, ahead of edit varimax's 95.4%, but physical heads also reconstruct better. Next test a learned sample-dependent shared dictionary against physical heads and exact rotations at matched reconstruction, then require held-out prediction, finite intervention fidelity, and shifted-token transfer.

That [learned shared-dictionary/group follow-up](CONTEXTUAL_COMPOSITION_SHARED_GROUP_RESULTS.md) now exists. A direct straight-through group gate fails the matched-reconstruction gate from sparse initialization, while an all-to-all membership becomes a flat model. A more stable unsupervised support-cluster fit creates genuine overlapping groups over a frozen learned 12-atom dictionary. On XOR held out from all featurizer fitting, two size-six groups retain 93.0% accuracy and .105 head error, near a physical top-three control's 93.2% and .108, but with worse finite logit edit-path error (.436 vs .409). Shifted-token transfer is also comparable. This particular group prior does not earn its extra complexity over physical heads; stop tuning membership penalties or group sizes here. A future group model needs a separately specified causal or compositional criterion and an independently verified operation-level target.

The [causal-prior and independent-seed follow-up](CONTEXTUAL_COMPOSITION_CAUSAL_PRIOR_RESULTS.md) tests that criterion on the same frozen attention architecture. A finite key-bit edit-path objective fitted only on primitive/AND queries improves a 12-atom flat featurizer's XOR edit-path MSE from .317 to .288 on five new model seeds, with a 5% seen-query reconstruction gate. Yet a physical-head gate calibrated on the same seen-query validation head error keeps ~4.24 heads and reaches .172 XOR edit-path MSE and 98.7% accuracy, versus the causal featurizer's 96.0%. The physical control wins in all five independent seeds; its held-out reconstruction error falls while the learned dictionary's roughly doubles. This is a transfer failure of the data-fitted dictionary, not proof that individual heads are canonical semantic features. Stop tuning this toy; seek operation-level ground truth and multi-family causal holdouts, then transfer the fidelity/legibility separation to the 546M model.

The [546M rare-key transfer test](FLAGSHIP_RARE_KEY_METRIC_TRANSFER_RESULTS.md) then checks the frozen uniform/mixed/corpus pre-RoPE rank-96 cuts on naturally occurring rare versus common keys. At the exact first-layer output, 128 documents per band/corpus and three cut-fit seeds give mixed the lowest rare-key error in all nine seed×corpus cells, while corpus-only has the lowest common-key error in all nine. A new 64-document rare-only full 18-layer panel gives mixed lower KL than corpus-only on all three corpora with paired bootstrap intervals below zero; mixed versus uniform is unresolved on FineWeb/WikiText and favors mixed on TinyStories. This supports the mixed cut as a broad fidelity prior, not a semantic canonicalization. Single-key downstream effects are tiny; the next legibility gate needs multi-head or behavior-level selective predictions at matched post-projection dose.

The [exact first-layer observable construction](EXACT_ATTENTION_OBSERVABLE_RESULTS.md) makes the weight-only reference precise. The summed token interaction tensor is uniquely determined by isolated length-one and length-two attention outputs, or computed directly from weights; its query-vs-(key,output) matrix has a canonical rank under a specified flattening. A 31-token, width-four, two-head CPU example has exact rank ten, saturating the symmetric-square query bound, while branch rescaling and head splitting leave the complete tensor unchanged. This is a useful function-level quotient before choosing data weighting or a sparse gauge, not a physical graph identity claim. Zero-support token pairs remain unidentified by dataset-only fitting, so weight-informed completion and held-out pair/behavior tests are essential.

The [later-head matched-dose gate](LATER_HEAD_MATCHED_AXIS_DOSE_RESULTS.md) applies the second-stage legibility comparison to two already task-selected 546M heads. Frozen effect-sparse and edit-sparse rank-four axes outperform task-selected random-rotation axes on all 38 bracket/agreement held-out prompt pairs after matching each finite patch's post-output-projection residual dose; the effect-axis mean margin-response ratios are 1.084 and 1.057. Weight sparsity loses to the random controls. This is a positive but modest **task-trained** coordinate result, not new-task discovery or physical necessity. The next run should withhold whole task families from selection or test operation-level graph recovery in independently trained toys.

The [answer-free matched-dose follow-up](LATER_HEAD_ANSWER_FREE_MATCHED_DOSE_RESULTS.md) reuses the same gate with rotations fitted to two separate 800-edit panels and no answer labels. When bracket/noun-number edits are included, raw edit PCA and edit L1 beat strongly task-selected random axes on all 38 held-out cases in both fit panels, with mean response-per-dose ratios around 1.05. Whole-type exclusion removes that advantage: agreement falls to .69/.72 for PCA and .16/.11 for edit L1, while bracket PCA is approximately tied or worse and bracket L1 falls to .50/.58. This isolates the useful prior as answer-free **but coverage-dependent** input intervention structure. The task-selected heads and designed edit categories prevent a genuine new-task discovery claim. Stop local gauge tuning here and move to task-blind head/operation selection or known-graph toy recovery.

The [active graph-discrimination control](ACTIVE_GRAPH_INTERVENTION_RESULTS.md) sharpens the physical-reuse boundary. In an exact same-external-function two-head toy, ranking feasible internal zero patches by the fitted candidates' Gaussian KL raises one-patch graph-selection accuracy above random, input-norm, and high-unpatched-activity selection across two true graphs, three noise levels, and candidate pools of 64/512. The active-versus-high-activity one-patch advantage is 5–14 percentage points with paired intervals above zero in all twelve cells; a held-out half-strength patch on shifted inputs checks prediction. The candidate graphs, factors, and intervention address are supplied, so the next challenge is candidate-operation discovery, not another function-only gauge.

The [trained physical-node alignment test](SHARED_PAIR_PHYSICAL_NODE_ALIGNMENT_RESULTS.md) compares the three strongest readable product nodes from the dense/paired shared-pair toys with the six exact physical head-pair terms feeding each model's quadratic readout. Across twenty trained models, each node has a large component outside the **entire** physical-pair span (mean norm fractions .744 dense/.749 paired), even though the six semantic nodes' outside-span pieces nearly cancel in their sum and the retained rank-three function preserves all five task decisions. This is a direct trained-model example of function-level legibility without physical node identity. The next high-value gate is distribution-weighted realizability of semantic-node edits by feasible internal patches, not more ordinary-function sparsification.

The [data-equivalent node-edit follow-up](SHARED_PAIR_DATA_EQUIVALENT_NODE_EDITS_RESULTS.md) tests that gate on 20 trained models and fresh task sentences. A six-term physical-span fit on seen combinations leaves .518/.525 relative MSE for readable nullspace nodes on paired/dense codings; an executable three-head attenuation leaves .594/.600. Letting the span see the rare combination barely changes full-test error. Two semantic products are partially approximable, but the A–B edit remains at .903/.944 relative error and worsens on the rare combination. Thus the distribution-aware metric is worthwhile, but it does not turn this legible external product graph into the tested physical head-pair graph. The next candidate set should include QK-branch and within-head edits with whole-combination and edit-strength holdouts.
