# Highest-information experiments for reverse engineering

## Where the evidence already stands

The existing experiments rule out both simple extremes.

- Native neurons are sometimes privileged, especially late in the network, but reusable computation is usually cleaner in a learned subspace than in individual neurons.
- A small per-example circuit is not automatically semantic: fixed-task width sweeps found increasingly privileged axes without reliable same-rule reuse.
- Arbitrary facts do form signed constraint networks, but the high-fit AdamW transformers have broad, weak, nearly orthogonal coupling rather than the taut web seen in the saturated one-layer memory.
- Every tested semantic edit has a low-dimensional local trajectory, but a subspace fitted to one word pair usually transfers poorly to another pair from the same named family.
- The 500M tensor model has different depth regimes: early predicate-like structure, a broad context-routed middle, and a compact output-role-dominated tail.

The main unknown is therefore not whether the model is “monosemantic” or “in superposition.” It is which explanatory object is stable, causal, reusable, and sufficient in each regime.

## Update after the computation-to-interface experiments

Several items below have now been partially executed. The controlled grammar
factorial shows that late MLP write/effect spaces can align across independent
initializations while early channel coordinates do not, and that this alignment is
relative to the task's random answer codebook. On the independent 12-layer language
model, a FineWeb-derived late write backbone transfers to WikiText and TinyStories,
but its required rank ranges from 32 to hundreds depending on the token edit and
fidelity target. A stable native channel ranking strongly beats random sets, while
one scalar direction and a linear coefficient router fail on the hardest OOD case.

The highest-information next tests are now:

1. **Upstream-to-backbone completeness.** For several natural operations, predict
   the shared-backbone coefficients from weight-derived bilinear/attention paths on
   blinded contexts. This tests whether the remaining difficulty is routing rather
   than span discovery.
2. **Real-model identifiability.** Train or obtain a matched independent checkpoint
   with the same tokenizer and objective, then compare native channels, write
   subspaces, and complete effects. The grammar result predicts increasing
   cross-run invariance toward the output interface.
3. **Backbone versus semantic tail.** Hold out whole operation families while fitting
   the shared basis. Ask whether the common backbone transfers and whether the
   residual tail is operation-specific, corpus-specific, or simply higher rank.
4. **Blind causal prediction.** Freeze the basis, native bank, and coefficient rule;
   predict full-distribution changes for unseen edits before running interventions.

## Update after exact-router and sparse-grammar tests

Item 1 has now been resolved at the final MLP boundary for one shared backbone.
A linear router fails OOD, while the exact native quadratic map restricted to a
natural rank-32 input space succeeds. Joint diagonalization falsifies an
independent-axis explanation; a replicated 32-edge graph gives a much shorter
causal program and reconstructs natural writes. The next tests should attack the
remaining boundaries, not repeat final-layer span discovery:

1. **Upstream coefficient provenance.** Starting from the 32 natural coordinates,
   use exact attention QK/value decompositions and integrated JVPs to predict each
   hub coordinate and pair activation one or two layers earlier. Freeze source
   selections before OOD evaluation.
2. **Cross-checkpoint graph identifiability.** Repeat the natural-subspace → joint
   tensor → sparse-edge pipeline on a matched independent checkpoint. Compare
   finite functions after Procrustes/functional alignment, not raw edge numbers.
3. **Tail decomposition by behavior.** Keep the shared 32-edge backbone fixed and
   factor only the orthogonal residual needed by period, comma, and common-word
   edits. Test whether tails form semantic modules or a smooth capacity frontier.
4. **Blind natural prediction.** Before intervention, predict which held-out sites
   incur the largest KL and which next-token margins change sign when selected
   edges are removed. This tests example-level completeness rather than aggregate
   reconstruction.

## Ranked experiments

### 1. Mechanism-transition factorial

**Question:** What changes a diffuse factual network into a taut KKT-like web?

Hold the task, data, architecture, factual accuracy, and edit scale fixed. Vary one candidate cause at a time:

1. optimizer: SGD versus AdamW;
2. slack: facts per parameter / width;
3. depth and residual paths: one layer versus four, residual versus no residual;
4. normalization: normalized versus unnormalized inputs.

Measure margin-gradient coupling, finite collateral under a one-logit normalized edit, effective support, margin slack, and KKT stationarity fit. The key endpoint is not training loss. It is whether the same learned facts become more mutually load-bearing.

**Why first:** It directly connects the original silence-code result to the transformer results and resolves the largest current conceptual confound.

### 2. Frozen causal-transfer tournament on the 500M model

**Question:** Which representation predicts a computation on genuinely unseen contexts?

For the same discovery examples, construct four candidate explanations:

- native neurons;
- an SAE or supervised sparse feature basis;
- tensor/spectral subspaces;
- a local weight-derived tensor/Jacobian program.

Freeze every direction, mask, coefficient rule, and threshold. Evaluate on unseen FineWeb documents and on WikiText/TinyStories. Score signed intervention-effect prediction, final-distribution KL fidelity, support stability, and description length. Include matched random rotations, frequency-matched token controls, and equal-rank/equal-budget controls.

**Decisive outcomes:**

- Native neurons win and transfer: a neuron-aligned regime.
- A rotated/learned subspace wins: distributed but reusable computation.
- Only per-example Jacobian programs win: context-specific routing/local charts.
- Nothing predicts finite effects: the proposed ontology is missing a causal variable.

### 3. Spectral outlier versus bulk causal separation

**Question:** Are the random-matrix outliers reusable circuits while the bulk carries addresses, corrections, or noise?

Use the same held-out transfer protocol to compare cleaned spectral spikes, bulk-matched modes, and sign/pairing nulls. Ablate or replace equal-energy subspaces and predict both local writes and downstream distributions. Test semantic generalization and exact-string/factual recall separately.

This turns the power-law-plus-spikes observation into a causal division-of-labor hypothesis instead of using a scree plot as evidence by itself.

### 4. Coordinate-identifiability experiment

**Question:** What part of an explanation is real rather than a convenient basis choice?

Train matched seeds and perform legal function-preserving symmetries where available. Also refactor or retrain compact substitutes to the same behavior. Compare stability of:

- individual neuron identities;
- semantic subspaces;
- signed downstream effect spaces;
- predicted intervention matrices.

If neuron labels vary but effect spaces recur, the canonical object is an equivalence class of programs, not a neuron. If even effect spaces vary while behavior is fixed, the explanation is underidentified at that level.

### 5. Blind circuit-completeness challenge

**Question:** Can the proposed reverse engineering actually make new predictions?

Hide a set of prompts, interventions, and small parameter edits. Using only discovery data and weights, require the analysis to predict:

- the sign and size of margin changes;
- which unrelated prompts suffer collateral effects;
- which semantic patch transfers;
- the final logit/distribution change;
- which component becomes necessary after another is removed.

Reveal the answers once. Report calibration and failures. This is the strongest guard against persuasive post-hoc stories.

## What not to prioritize now

- More top-token lists without frozen causal transfer.
- More low-rank reconstructions without downstream behavioral tests.
- A single monosemanticity scalar that mixes sparsity, axis privilege, semantic reuse, and causal sufficiency.
- A neuron viewer before the ontology and transfer tests are stable.
- Treating power-law outliers, SAE sparsity, or small circuit size as sufficient evidence of a human-readable circuit.

## Practical order

Run experiment 1 to establish the controlled mechanism transition. In parallel conceptually, specify experiment 2 as the real-model benchmark. Use experiment 3 as a preregistered candidate within that benchmark, not as a separate fishing expedition. Experiments 4 and 5 then determine whether the best explanation is canonical and complete enough to count as reverse engineering.

## Updated priority after the Layer-6 Jacobian result

The local causal algebra is now validated. More generic finite-intervention
checks on the same route have diminishing value. The highest-information next
tests are:

1. **Semantic mode challenge.** Freeze the leading FineWeb Jacobian modes,
   choose their best simple semantic labels on a discovery split, and evaluate
   labels and signed finite effects on disjoint FineWeb, WikiText, and
   TinyStories. Compare individual PCA modes with random rotations of the same
   subspace. This determines whether spectral outliers are named verbs or merely
   reusable geometric corrections.
2. **Cross-checkpoint functional alignment.** Repeat the Jacobian spectrum in an
   independently trained tensor model and align modes by causal action rather
   than coordinates. A reusable ontology requires effect-space recurrence.
3. **Synthetic rule/memorization sweep.** Apply the same Jacobian-chart and late
   sparse-grammar measurements across known rule fractions and capacities. Test
   whether reused grammar reduces effective Jacobian rank or increases spectral
   outlier strength, while arbitrary facts increase the bulk or constraint-web
   dependence.
4. **Attention-inclusive analytic chain.** Add exact QK/value derivatives to the
   already exact MLP chain. The current omission costs only about 2–3% for this
   route, but closing it is necessary before calling the full local computation
   reverse engineered.
5. **Blind behavioral challenge.** Use only weights and unperturbed activations
   to predict one-sided distribution changes for held-out sites, including edits
   chosen after the analysis is frozen. Preserve the central-difference bug as a
   mandatory regression test.

The attention-inclusive local challenge now passes at about 0.996 fidelity.
Accordingly, the two most informative remaining experiments are no longer more
local derivative checks:

- **Cross-checkpoint effect-space alignment:** determine whether leading
  Jacobian subspaces and the late sparse grammar recur after independent
  training, even when their coordinates rotate.
- **Known-rule versus arbitrary-memory sweep:** determine whether rule reuse
  changes spectral outlier strength, Jacobian effective rank, late-graph
  sparsity, and constraint-web coupling in the predicted directions.

These distinguish a model-specific chart from a reusable ontology of neural
computation.

## Updated priority after the controlled grammar operator sweep

The known-rule versus arbitrary-memory sweep is now complete and has high
discriminating value. Pure rules use a compact early causal-operator family;
50–100% arbitrary load expands it by roughly an order of magnitude. The effect
survives identical prompts, bilinear gauge normalization, natural activation
covariance, softmax Fisher weighting, margin restriction, and high-confidence
held-out rule examples. Width reduces rank per channel but does not restore the
pure-rule implementation. Both regimes converge to a compact output-role tail.

The highest-information next experiment is now **recursive tensor-program
compression**:

1. derive the exact Layer-3 answer-interface Jacobian as a positive control;
2. recursively add Layer-2 and Layer-1 bilinear plus QK/value derivatives;
3. factor only fixed weight tensors using matched Tucker, tensor-train, and sparse
   bilinear budgets;
4. retain explicit state contractions/gates rather than regressing one smooth
   global map;
5. freeze the factorization and predict complete held-out operators plus finite
   effects across rule loads, templates, and seeds.

This directly distinguishes two remaining hypotheses. If pure and mixed models
share one small tensor program with different state contractions, high Jacobian
rank is a local atlas generated by reusable computation. If mixed models require
larger tensor rank and description length and those extra factors are causally
necessary, arbitrary storage changes the implemented program itself.

The internal-controller negative result must remain a boundary condition: linear
PCA codes of even the full residual sequence and a top-16 quadratic expansion do
not predict the early mixed chart. A successful tensor method must beat those
frozen baselines rather than merely refit the same examples.

## Updated priority after cross-initialization recurrence tests

The native gate-bank, correct-margin, and global linear-conjugacy tests are now
complete. Individual channels do not recur across initializations, and a learned
state map does not transport the full local answer Jacobian. Pure-rule models
nevertheless show substantially more late gate-space recurrence than mixed
models, while previously frozen semantic edit families transfer causally across
pure models. The high-information target is therefore no longer unrestricted
model isomorphism.

The next experiment should be a **held-out operation-family tournament**. On the
same checkpoints, predeclare semantic edit families, fact-specific edits, and
random tangent directions. Fit one alignment without intervention outcomes,
then predict finite effects for held-out families and templates. Compare:

1. native channel permutation;
2. global linear gate map;
3. relation-conditioned linear maps;
4. exact weight-derived local tensor transport.

This directly estimates the domain over which each ontology is correct. A
global map winning only semantic families means reusable programs but no global
conjugacy; local tensor transport winning arbitrary tangents means exact local
reverse engineering without a canonical cross-run basis; failure on held-out
semantic families falsifies the claimed reusable program rather than merely
showing neuron labels differ.

The exact recursive factorization has now passed at machine precision. It
localizes most mixed variation to the two bilinear arms of the final downstream
block, with a smaller QK-pattern contribution and negligible local-value
contribution. The priority therefore narrows again:

1. **Gate-code factorization:** treat left/right preactivations and QK-pattern
   derivatives as the targets. Compare native channels, sparse bilinear
   dictionaries, tensor train, and matched random rotations under one frozen
   description-length budget.
2. **Finite path validation:** implement path-specific JVP or integrated-Jacobian
   replacements for `ML`, `MR`, and `Q`. The current mean-factor replacements are
   exact tangent recombinations but not literal finite module ablations.
3. **Cross-seed functional alignment:** align gate programs by their induced
   residual operator, not channel number, and demand prediction on held-out facts.

A winning gate model must predict the complete operator and finite answer effect;
decoding the randomization flag or answer alone is not sufficient.

## Updated priority after the operation-family tournament

The broad tournament is complete. A global distributed gate map and a
weight-quotiented write map both transport pure semantic edits; late generic
prompt-pair differences transfer much less. Mixed semantic effects partially
align, while arbitrary swaps do not align under native, global,
relation-conditioned, or write-quotient maps. Increasing sample count rather
than relaxing the causal threshold replicated these distinctions.

The next highest-information experiment is now **address-program prediction**
inside one mixed checkpoint. Cross-run nonrecurrence says arbitrary facts do not
share a canonical program, but it does not tell us whether each run uses:

1. a compact deterministic address router into a shared write dictionary;
2. many fact-local tensor contractions with no short selector;
3. a constraint-coupled parameter web whose activation routes are secondary.

For held-out learned arbitrary facts, freeze candidate selectors derived from
ID digits, full token-state tensor contractions, attention QK addresses, and
nearest-neighbor gate codes. Predict the exact active residual write and finite
answer effect before intervention. Compare description length and OOD transfer
against per-fact lookup and shuffled-address controls. Then perform small
parameter edits along the predicted supporting constraints and require the same
model to predict collateral fact changes. This joins activation routing to the
KKT/gradient web instead of treating them as competing stories.

### Address-program prediction update

The first within-checkpoint branch is complete. The final normalized address is
compressible by a rank-48/96 activation subspace executed through the untouched
bilinear tensor, and the subspace learned on rules transfers to arbitrary facts.
The hypothesis that arbitrary memories occupy a separate final subspace is
falsified. Additive relation-answer prototypes also fail on arbitrary facts;
valid codeword swaps show that answer routing survives much better than the
exact eight-logit effect.

The next discriminating experiment should therefore move one boundary upstream:
predict or causally reconstruct how the model maps the subject digits, object
digits, and relation into a valid final codeword. Compare an exact recursive
tensor contraction, a sparse routed dictionary, and a per-fact lookup under one
description-length budget. Score (a) final answer, (b) exact address state, and
(c) exact finite effect separately. Then connect the winning activation route
to layerwise margin-gradient edits; neither route prediction nor a tangent
constraint graph alone establishes where the arbitrary association is stored.

### Final-gate positive control for upstream retrieval

Although 192 native channels participate, a seven-dimensional answer-centroid
span preserves 0.96--0.97 of arbitrary-fact causal effects and decisively beats
100 label-permutation nulls. The next upstream model should therefore be judged
by whether it predicts the *continuous coordinates inside this frozen answer
quotient*, not merely the answer label. Label decoding would miss the remaining
context calibration and could falsely declare retrieval solved.
