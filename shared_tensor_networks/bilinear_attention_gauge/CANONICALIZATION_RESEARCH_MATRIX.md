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
