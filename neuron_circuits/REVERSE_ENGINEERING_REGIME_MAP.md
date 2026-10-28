# A regime map for reverse engineering memorization and reusable computation

## Bottom line

The useful picture is not a spectrum with “monosemantic neurons” at one end and
“memorization webs” at the other.  The experiments support several partly
independent coordinates:

1. **semantic transfer** — does the same intervention implement the same change
   on unrelated examples?
2. **native-axis privilege** — is that transferable change unusually sparse in
   the trained neuron basis compared with random rotations?
3. **instance-specific state** — after accounting for known semantics, is a
   stable fact-specific activation pattern left over?
4. **constraint coupling** — which other predictions move under a parameter edit,
   with what sign and effective support?
5. **behavioral fidelity** — did the model actually learn both the behavior and
   the interface on which the mechanistic comparison depends?
6. **extractor identifiability** — is the reported circuit a stable property of
   the model, or can a tiny metric perturbation make the explanation algorithm
   jump between equally scored but causally different supports?
7. **task-codebook dependence** — does the representation recur across runs only
   after holding fixed arbitrary label, vocabulary, or output conventions?

A model, layer, or subspace can be high on more than one coordinate.  A coherent
output role can coexist with a dense web of input-side factual constraints.  A
distributed semantic algorithm can have weak neuron alignment.  A local
gradient web can accurately predict edits even when the underlying computation
is a reusable rule rather than memorization.

## New controlled result

The experiment in this repository trained one-, two-, and four-layer decoder
transformers on an autoregressive synthetic language.  Shared rules and fixed
arbitrary exceptions used the same names, prompts, answer histogram, and model
interface.  Both SwiGLU and bilinear MLPs were tested.  The decisive depth-four
conditions were repeated over three independently generated datasets and model
initializations.

The frozen protocol is in
[`artifacts/grammar_regimes/preregistration.md`](artifacts/grammar_regimes/preregistration.md).
The aggregate data and figure are in
[`artifacts/grammar_regimes/summary/summary.md`](artifacts/grammar_regimes/summary/summary.md)
and
[`artifacts/grammar_regimes/summary/regime_summary.png`](artifacts/grammar_regimes/summary/regime_summary.png).

### Behavioral finding

Pure-rule depth-four models generalized the rule to held-out pairs at 99.7–99.9%.
With 50% arbitrary exceptions, models still recalled the training table at about
98–99%, but accuracy on held-out *unrandomized* rule pairs fell to about 43% for
both architectures.  Fully arbitrary tables were also learnable.  The mixed task
was therefore not an arithmetic midpoint: shared rules and exceptions competed.

Two of six mixed seed/architecture cells achieved near-perfect trained-template
recall but failed the preregistered unseen-template gate.  They remain learning
failures and were not promoted into matched-fit representation evidence.  Their
learning curves peaked and then became more template-specific, so simply running
longer was not supported by the observed trend.

### Semantic transfer and neuron alignment

For direct grammar rules, a discovery set supplied a mean post-product activation
change for changing one latent value.  That frozen change was applied to unrelated
entity names and an unseen template.  Selection and evaluation identities were
disjoint.

In the six pure-rule models, full final-layer patches changed the recipient to the
counterfactual answer with mean accuracy 0.66 (SwiGLU) and 0.72 (bilinear).
Keeping the 32 largest native channels retained almost all of this effect: 0.63
and 0.62.  Keeping 32 random native channels gave 0.002 and 0.001; keeping the 32
largest coordinates after a fixed random rotation gave only 0.032 and 0.037.

This is causal native-axis privilege.  It supports **neuron-aligned reusable
computation** in the pure grammar condition.  It does not prove that every selected
neuron is a single human concept, or that the 32 channels are a minimal circuit.

At 50% arbitrary exceptions, full-patch transfer fell to 0.15 (SwiGLU) and 0.26
(bilinear).  Native-32 patches fell to 0.10 and 0.14, only slightly above rotated-32
at 0.08 and 0.13.  The model could often answer the tested base and counterfactual
prompts naturally, yet the transformation estimated from other names no longer
ported cleanly.  This is evidence against a shared neuron-aligned rule being the
dominant implementation in the mixed regime.

### Stable instance state

The activation audit first selects a simple semantic explanation on one trained
template and evaluates it on unseen templates.  It then subtracts that explanation
before asking how well an exact fact's residual activation transfers across
templates.  This correction matters: unresidualized template stability falsely
calls a deterministic rule state an address code.

At the final layer, median residual fact stability was 0.24–0.47 for pure rules,
0.85–0.86 for mixed tables, and 0.85–0.91 for fully arbitrary tables.  The nonzero
pure-rule remainder is a warning that the semantic vocabulary is incomplete, not
a license to name every residual as memorization.  The large and replicated gap is
the defensible result.

### Constraint networks

For prediction margin `m_i`, the local effect of a normalized edit along
`g_i = grad_theta m_i` is predicted by the signed Gram matrix
`K_ij = g_i^T g_j`.  Finite parameter edits validate this causal local model in
all regimes: at the standard edit scale pooled correlations are 0.98–1.00.
Wrong-source controls fail, and larger edits show the expected nonlinear boundary.

Crucially, this does **not** identify memorization.  Pure rules also have a valid
constraint network.  What changes is its geometry: early-block effective support
is about 11–13 for pure rules and 22–25 for mixed/arbitrary tables.  Arbitrary
edges are broader but often individually weaker.  This reproduces the earlier
factual-capacity result that “stronger web” and “broader web” are different claims.

KKT is the right term for the single-layer margin interpretation: under the
relevant homogeneous-margin assumptions, a stationary max-margin solution can be
written as a weighted sum of active constraint gradients.  The finite-edit result
here validates the local signed interaction map; it does not establish global KKT
optimality for AdamW transformers.

## Regimes to expect

| Regime | Behavioral signature | Representational signature | Best first tool |
|---|---|---|---|
| Underfit or interface-brittle | poor recall or unseen-template transfer | uninterpretable comparisons | fix/measure learning first |
| Native semantic features | counterfactual rule transfer | native sparse patches beat rotations | neurons/channels plus exact patches |
| Privileged distributed predicate bank | related edits reuse native axes, but frozen high-fidelity banks are broad | native basis beats rotations while S90 requires much of the layer | neuron atlas for atoms + family frontier + tensor routing |
| Spatially split feature code | a token property is stored locally and later consumed elsewhere | distributed-prefix banks transfer, but query-position banks do not | position-indexed banks + source factorial + attention paths |
| Generic transport skeleton with state-selected assistants | the same small component family carries many edits, but exact membership varies by input | frozen heads beat almost all fixed subsets; per-state heads close the remaining gap | discovery-frozen head bank + exact QKOV contraction |
| Conditional query readout | upstream transport is stable, but neither native banks nor fixed write subspaces transfer at the receiving position | per-example native oracle succeeds while rank-24 frozen spans remain weak | bilinear state-conditioned map from transported value to product/write |
| Dynamic native alignment | semantic identity does not fix neuron identity, but each instantiated computation is sparser in native products than dense rotations | native per-example frontier beats rotations; fixed family bank fails | exact product rule + state-conditioned native ranking |
| Algebraic verb with unstable operands | a native product has a clean operation, but which factor carries the semantic edit changes with context | asymmetric cross terms recur above null; factor-side transfer weakens with depth | symmetry-aware factor-role audit + OOD side permutation |
| Distributed semantic computation | rule transfer remains high, native advantage weak | low-dimensional or sparse learned subspace | subspace patching, sparse dictionaries |
| Instance/address memory | seen recall high, unseen arbitrary facts at chance | stable residual fact state, weak semantic transfer | fact ledger plus instance-conditioned probes |
| Broad constraint storage | edits have many signed collateral effects | large gradient-kernel support | layerwise gradient sketches and finite edits |
| Additive calibrator | behavior varies along a nearly fixed write direction | low-rank removal is necessary and specific | behavior-conditioned direction + removal |
| Conditional router | effect depends on context and source/recipient state | a probe decodes it, but fixed-direction removal fails | query/gate/operator-conditioned analysis |
| Stable operator, unstable support | smooth local Jacobian; discrete explanations jump | canonical projectors agree while greedy neuron sets differ | symmetry-aware eigenspaces plus finite validation |
| Operator with state-supplied content | one route transports many kinds of changes | head/factor route transfers; semantic activation direction rotates | factorized operator + state contraction + value analysis |
| Codebook-relative output interface | upstream evidence changes but late action is shared | effects align within one output convention, not an unseen random relabeling | write/effect quotient plus same-codebook replication |
| Compact cooperative write module | low-rank output effect, multi-term execution | learned write subspace transfers; individual bilinear terms fail; fixed native subset beats random | write SVD + exact bilinear expansion + channel frontier |
| Local atlas | each example has a compact chart but no fixed global route | local subspaces work; frozen selectors fail in tails | chart transitions and held-out routing |
| Redundant circuit | named components are causal but individually unnecessary | cumulative ablation differs from single ablation | groups, paths, and redundancy curves |

These regimes can coexist inside one layer.  The practical unit of explanation
should be whichever object survives out-of-distribution prediction and causal
intervention: sometimes a neuron, sometimes a subspace, an attention readout, a
sparse operator, or a signed constraint neighborhood.

## Computation-to-interface and codebook factorial

The newest controlled assays refine the pure-grammar headline. At every tested
depth, most of a semantic patch's downstream effect lies on a calibrated generic
answer direction: vector Shapley projection assigns 0.93–0.95 to that answer
component. Upstream nonlinear interaction is larger, so this does not relocate
the rule algorithm into an answer neuron. It says semantic evidence is converted
into a shared actuator.

A prospective cross-model test then separated initialization from task convention.
Global activation maps that saw every relation transferred same-rule edits strongly,
but this was vulnerable to alignment leakage. Leave-one-relation-out raw-channel
alignment failed. After quotienting bilinear channel symmetries into residual
writes, independently initialized models sharing one manifest acquired an
adjudicative Layer-3 equivalence: held-out R2 0.624 mean/0.423 minimum, same-rule
effect fidelity 0.883, wrong-rule fidelity 0.213, and target-answer fidelity 0.898.
Earlier layers remained non-adjudicative.

Changing the random answer permutation destroys this unseen-relation equivalence:
Layer-3 write R2 becomes -0.490 and mapped fidelity 0.103, while each target model's
own answer interface remains at 0.906. The late computation is consequently
canonical only relative to the learned codebook. This is expected, not a defect:
the random permutation is part of the task the network implements.

## Independent natural-language boundary result

On the independently trained 12-layer tensor language model, a natural newline
operation shows both the usefulness and limit of the same ontology.

- An unpaired natural-newline write prototype is nearly collinear with a paired
  token-to-newline write edit (cosine 0.986) and decisively beats a period control.
- A single fixed ray works on held-out FineWeb and TinyStories but falls to 0.376
  fidelity on WikiText. Oracle per-example rescaling reaches only 0.464, so the
  missing variable is direction, not merely amplitude.
- A FineWeb-derived write subspace transfers: the 0.80 fidelity threshold is
  reached by rank 4 on FineWeb, rank 8 on TinyStories, and rank 32 on WikiText.
  Equal-rank random subspaces fail. A linear state router remains insufficient on
  WikiText, separating span discovery from coefficient routing.
- The exact bilinear identity decomposes the edit into base×edit cross terms and an
  edit² term with relative error below 8e-7. Neither term is causal alone; their
  sum reaches 0.81–0.88. The stronger preregistered cancellation claims fail,
  though moderate anti-alignment and pairing dependence remain.
- A discovery-fixed native subset executes most of the compact module: 32 channels
  reach 0.69–0.75 fidelity, 128 reach 0.77–0.84, random subsets are near zero, and
  per-example oracle selection adds only 0.01–0.03.

This gives a concrete middle ontology between monosemanticity and opaque
superposition: a low-dimensional, reusable effect-space verb, computed by a stable
bank of privileged native bilinear channels whose contributions cooperate and
partly cancel. Individual channel meaning and upstream semantic evidence remain
separate interpretation problems.

The boundary circuit is not representative of every token edit's compression
level. Across newline, period, comma, and ` the`, target-specific rank-32 bases
overlap strongly and cyclic wrong-operation bases work nearly as well. One pooled
rank-32 basis and one pooled top-128 bank match all target-specific counterparts,
so the late layer has a shared edit backbone rather than four separate modules.
But the tails differ sharply: mean minimum rank for 0.80 fidelity grows from 32
to 213, 299, and 384 across those four operations. At 0.90, some edits require
512–768 residual dimensions and 2,048 of 3,072 native channels. The correct
quantity is therefore a behavior-and-fidelity capacity curve, not “the number of
neurons in the final computation.”

## New regime: sparse relational grammar on a natural state manifold

The newest final-layer experiment resolves the earlier linear-router failure.
At the final pre-MLP boundary, pooled write coefficients are not arbitrary:
they are an exact quadratic function of the state fixed by the native bilinear
weights. Restricting those quadratic forms to 32 ordinary-state PCA directions
gives a compact OOD-transferable program. A stable joint rotation does **not**
make the forms diagonal. Independent scalar coordinates recover only 0.35–0.57
fidelity; 32 selected pair terms recover 0.72–0.81 and reproduce the full
compact program.

| Object | What succeeds | What it means |
|---|---|---|
| Scalar axes | Category decoding survives many rotations; canonical axes do not beat random rotations | Information is in the subspace, not uniquely neuron/axis aligned |
| Pair graph | 32/496 edges replicate across selection panels, operations, and corpora | A sparse relational computation is privileged within that subspace |
| Full native tensor | Exact algebra but much larger ambient representation | Mechanism is known even where its shortest useful description depends on the natural manifold |

The graph also survives a natural-computation test. Replacing only the exact
pooled component of the native final-MLP write with the graph reconstruction
recovers 92–94% of the component's KL effect, including non-newline sites. That
is stronger than top-token coherence: it is a finite distributional
reconstruction while all orthogonal native writes are preserved.

The correct label is therefore neither “monosemantic neurons” nor “unstructured
superposition.” It is **distributed coordinates with a sparse reusable
interaction grammar**. Human semantics describe several hubs and edges, mostly
around boundary/punctuation structure, but the causal program—not the label—is
the primary evidence.

## Connection to the 500M FineWeb tensor transformer

The original version of this section only synthesized already-recorded 545.9M
results. The current repository now also contains a new sequence of exact
Layer-8 tensor/Jacobian experiments on the model. They materially change the
ontology:

- Complete singleton and pair route-response tensors reconstruct three-layer
  attention routes essentially exactly in bulk, including blind FineWeb and
  TinyStories replication.
- The rare apparent catastrophic failures were not a finite nonlinear regime.
  Tiny higher-order residuals made a discontinuous greedy 16-mode extractor
  jump between causally different supports. On a fixed support reconstruction
  fidelity is 1.000 median, and each support's complete analytic tangent
  predicts its own 0.1%-norm finite effect at 1.000 median and 10th percentile.
- A symmetry-aware rank-16 sensitivity projector is stable exactly where the
  neuron set is not: exact/reconstructed overlap is 1.000 median and 0.997
  minimum, with 0.999/0.992 median/10th-percentile finite fidelity.
- Route and full-native projectors overlap around 0.94–0.95 and are 16–22 times
  more causally sensitive than random subspaces, but their active finite effect
  can differ. The missing contraction matters: cosine between `P_route x` and
  `P_full x` predicts effect fidelity with Spearman 0.867. Literal removal
  reproduces the result, ruling out norm amplification as the explanation.
- Natural lexical edits reveal a reusable Layer-4 head-specific broadcast route,
  but not clean sentiment/gender/tense routes. In a prospective 17-new-pair
  test, correct-family advantages are only 0.014/0.016, while a single global
  route is as good or better. The operator is reused; value/state supplies the
  content.

These new results sit alongside the earlier recorded examples:

- The 500M quantization audit found that one scalar attention mixing parameter was
  a single point of failure.  Removing that parameter from the quantization sweep
  changed normalized-margin forecasting from weak (rho about 0.21) to much stronger
  (about 0.49).  This is a concrete warning that global perturbations can measure a
  bottleneck rather than distributed factual structure.  See
  [`../memorization/results/t8_500m_audit.jsonl`](../memorization/results/t8_500m_audit.jsonl).
- Natural induction localizes to several named heads, led by L5H5, but ablating the
  top four removes only about 9.6% of copying.  That is component alignment plus
  redundancy, not a single necessary “induction neuron.”  See
  [`../reverse_engineer/tensor_language/RESULTS.md`](../reverse_engineer/tensor_language/RESULTS.md).
- Block 17 contains a behavior-conditioned rank-one frequency-calibration direction
  that is necessary and specific but only 33% sufficient by itself.  The same
  low-rank method fails on conditional newline routing.  This cleanly separates an
  additive low-rank regime from a conditional router regime.
- Full bilinear-MLP Jacobian clustering reduces largely to input cosine on real
  activations, while a bilinear-secant sparse operator dictionary reconstructs the
  layer-6 operator collection with FVU about 0.12 using 32 active atoms.  A matched
  dense low-rank control is weaker in the layers where genuine sparse adaptation is
  strongest.  This is precisely the “distributed but mechanistically structured”
  case that neuron inspection misses.  The authoritative synthesis is
  [`../reverse_engineer/tensor_language/jacclust/SUMMARY.md`](../reverse_engineer/tensor_language/jacclust/SUMMARY.md).
- In attention, the useful explanatory object is often a restricted gate/readout:
  query features and the two-QK factors predict causally different operations more
  reliably than the full contaminated Jacobian.  This supports choosing the
  explanatory object by the module's algebra rather than insisting on one universal
  neuron-level representation.

The synthetic result and the 500M evidence agree on the central point: real models
should be expected to contain a mosaic of aligned features, sparse distributed
operators, conditional routers, calibrators, redundant circuits, and broad local
constraint networks.

## A concrete workflow for a real model

1. **Choose one behavior, not “meaning” in general.** Build a ledger with positive,
   negative, and counterfactual examples.  Split discovery identities/templates from
   transfer identities/templates.
2. **Pass a behavioral gate.** Require intact accuracy, counterfactual sensitivity,
   and interface/template robustness before interpreting representation metrics.
3. **Locate the causal component.** Measure exact ablations of residual, attention
   outputs, and post-product MLP channels.  Use both promotion and suppression effects.
4. **Test native axes against basis controls.** Compare a fixed-size native patch with
   random native channels, random rotations, and a learned subspace at equal rank or
   retained norm.  Decodability alone is not axis privilege.
5. **Test semantic transfer.** Freeze the patch or edit on discovery contexts and ask
   whether it performs the same transformation on new tokens, entities, syntactic
   frames, and corpora.
6. **Build a local constraint map.** Sketch per-example margin gradients by layer,
   regress out shared answer/token/input features, and validate predicted collateral
   effects with finite edits.  Treat support as breadth, not semantic unrelatedness.
7. **If neurons fail, change the explanatory object.** For bilinear MLPs use
   gate-restricted forms or secant/operator dictionaries; for attention use query,
   value, QK-factor, and head-output objects.  For additive biases try low-rank
   behavior-conditioned directions.  For conditional effects use state-dependent
   operators rather than fixed directions.
8. **Measure redundancy and necessity separately.** A component can be causal without
   being necessary.  Report single, cumulative, and matched-random ablations.
9. **Demand a fresh prediction.** A mechanism graduates only when it forecasts an
   unseen patch, edit, collateral set, or corpus transfer with a frozen selection rule.

For a 500M model, full per-example gradients are unnecessarily expensive.  Start with
32–128 behaviorally stratified positions, layerwise VJP/JVP sketches, and 4–8 finite
edits per candidate layer.  Activation collection can be streamed.  The existing
500M work shows that the heavier operator-SAE and causal head tests are feasible on a
16 GB GPU when performed one layer/head at a time.

## Red-team record

The result survived the following checks:

- The original modular grammar was rejected because high training accuracy coexisted
  with poor rule generalization; the simplified grammar and change are retained in
  [`artifacts/grammar_regimes/design_changes.md`](artifacts/grammar_regimes/design_changes.md).
- Interpretability comparisons require both recall and unseen-template gates; failed
  seed-2 cells remain reported rather than silently optimized away.
- Axis claims use 100 fixed random rotations plus random-native controls.  The causal
  patch difference, not a barely significant descriptive R2, carries the claim.
- “Fact address” was corrected to residual fact stability after discovering that the
  original score conflated semantic state with identity.
- Parameter-gradient predictions were tested with finite edits, a wrong-source null,
  and a larger edit scale.  Their success was explicitly demoted from a
  memorization-specific signature to a general local causal fact.
- Conclusions were repeated across three data/model seeds and two MLP architectures.
- Arbitrary held-out labels were never treated as learnable targets.
- The proposed cost-whitened tail certificate was tested blindly on new
  FineWeb/TinyStories and failed to reject any of twelve failures; it is retained
  as a falsified certificate rather than tuned after the fact.
- A tangent audit initially compared finite full-suffix interventions with a
  route-masked tangent. Passing controls exposed the mismatch; the audit was
  corrected to use the full analytic suffix Jacobian, with the error and
  unchanged thresholds recorded in the preregistration.
- Canonical subspaces were compared with the full native Jacobian and random
  subspaces, preventing internal exact/reconstructed agreement from being
  mistaken for native relevance.
- Borderline semantic-family transfer was followed by equal-data outside-family
  controls and a prospective test on seventeen frozen new lexical pairs. The
  stronger semantic-route claim failed rather than being rescued post hoc.
- Cross-model grammar transfer was first tested with all relations in the alignment
  panel and looked strongly positive. A leave-one-relation-out red team made the
  different-codebook result non-adjudicative; same-manifest retraining and
  symmetry-aware write alignment then isolated the valid late equivalence.
- The natural newline ray failed on WikiText even after its state-dependent scalar
  was predicted with R2 0.81. The claim was changed to a compact write subspace
  only after a frozen rank frontier and equal-rank random controls succeeded.
- The first analytic bilinear shuffled-base control evaluated an already-useless
  cross term alone. The corrected test retained the quadratic term, preserved the
  original failed threshold, and found a moderate 0.18–0.27 pairing effect rather
  than the predeclared universal 0.20 effect.
- The first Layer-0/1 source summary was discussed as though an all-position
  family-bank metric were a final-query metric. The audit now reports both scopes.
  Distributed-prefix semantic banks transfer, while prospectively frozen
  final-position family banks fail in three of four pair-clustered cells.
- The final-attention exactness gate remains formally failed at `4.33e-5` because
  the checkpoint's float32 contraction order missed the locked `2e-5` threshold.
  A separately labeled algebraic diagnostic recomposes at `3.99e-6`; thresholds
  were not rewritten after the run.

Remaining limitations are important: only direct grammar rules were used for the
causal transfer headline; the semantic feature vocabulary is incomplete; width was
fixed in the replicated grammar experiment; AdamW checkpoints do not test the
single-layer SGD KKT claim directly; the new 500M assays primarily characterize
local causal sensitivity rather than full natural-computation sufficiency; and
the generic broadcast result has been tested at one model checkpoint and layer.

For the first two layers of the 546M tensor model, the present best unit of
analysis is a conditional `head → factor → product-channel` route. Layer 0 has
a reusable head-7-centered transport anchor but changing product-channel
identity; Layer 1 mixes several heads with carried state. Accordingly, use
native neurons to explain an instantiated execution, but use heads, factor
weights, and state-conditioned routing rules when asking what transfers across
examples.

For finite lexical interventions, add a path test before concluding that a
failed endpoint Jacobian means discontinuous routing. In Layers 0/1, the base
JVP is accurate only for small doses, while a target-conditioned midpoint JVP
recovers nearly all finite native support. Use endpoint Jacobians for local
sensitivity; use one or more interior contractions when compiling a declared
large edit. Always compare with a wrong-target path and base-active channels,
because broad execution atoms can otherwise mimic target-specific routing.

Do not interpret successful finite-edit compilation as semantic abstraction.
Run both natural wrong-target and norm-matched off-manifold controls. Separate
three outcomes: endpoint behavioral fidelity, native-support overlap, and
cross-example semantic reuse. In the early tensor layers, fidelity is mostly a
generic smooth-path property, while native support is cleaner for natural token
edits; neither establishes a shared sentiment/gender/tense neuron code.
