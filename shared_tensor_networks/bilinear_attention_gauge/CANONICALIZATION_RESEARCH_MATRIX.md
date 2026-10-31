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
matched canonicalization comparison to later-layer 64-dimensional head
outputs, with native-model finite interventions and whole-task holdouts.
Head-output sufficiency does not yet prove necessity, source computation, or
a human-readable component; these remain separate validation gates.
A concrete position-to-position path hypothesis could separately
justify tracing earlier-position QK1-key effects. A
separate branch-level experiment can define causal response signatures
across multiple edit families and corpora, including QK1 or two-branch
perturbations, and optimize
condition-wise legibility only after reduction fidelity is gated on
disjoint corpora and downstream logits. The earlier query/key-form
factorial already rejected pair-level directional labels, so
selected-pair and full-model readouts must remain separate.
