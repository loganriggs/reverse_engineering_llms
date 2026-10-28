# Interpreting label-blind circuit motifs in Bilin18

## Bottom line

The model does **not** divide cleanly into a small global vocabulary of attention-head types under the weight metric tested here. It does contain reproducible **local operator motifs**.

Two motifs now have strong held-out semantic evidence:

1. **Recurrence routing:** read a prior occurrence of the current token.
2. **Immediate-left composition:** read the token immediately to the left.

These are best understood as neuron/head-aligned **verbs**, not stored object labels. Their semantics live primarily in the QK address: the recurrence heads put about 11× more address mass on prior exact matches than edge prevalence predicts, and the local-composition heads put about 20× more on distance-one edges. Their OV source-write norms are ordinary (roughly 0.84× and 1.00× enrichment respectively).

The initial broad **multiscale boundary** interpretation did not survive the same tests. Some individual heads are boundary-sensitive, but the frozen weight motif is not yet one coherent boundary family.

The separate 50-label supervised census is useful as an **external anchor set**, but it does not currently validate semantic names for the weight motifs. Thirty-two labels passed coverage and model-competence gates. With 50 held-out contexts per label, their head-level causal geometry is reproducible (`r = 0.507` across 25/25 splits), but frozen motif members show only a nonsignificant signed-profile trend over exact layer-matched controls (`+0.041` cosine, `p = 0.080`). Magnitude profiles cluster strongly, but a Gram-matched nonsense ontology does too. The motifs therefore share generic causal salience more clearly than human-label semantics.

This is compatible with Cammarata's mechanism-first picture, but it is not a complete test of recursive generalization. We have only the final checkpoint, so we cannot watch branches grow over training.

## What was discovered without labels

Each of the model's 162 native attention heads was represented by two symmetry-invariant random-function sketches:

- its bilinear `QK1 × QK2` address operation across relative rotary distances;
- its native value-bus-aware `OV` write operation.

No tokenizer strings, token categories, natural-text contexts, or proposed semantic tasks entered this stage.

The pairwise geometry replicates across independent sketches (`r = 0.812`), but forced global partitions do not: the best adjusted Rand index among `k ∈ {12,18,24,36}` was only `0.153`. Therefore the global clusters were rejected rather than labeled.

At a local scale, the top-three-neighbor sets replicate (median Jaccard `0.50`). Requiring a pair to be mutual top-three neighbors in both sketches produces 64 stable edges and 23 nontrivial components. Eighty-one of 162 heads are isolated under this strict threshold. The result looks like a local circuit ecology on a continuous geometry, not an exhaustive taxonomy.

## Architectural red team: the persistent value bus

The first local graph was partly confounded by the architecture's reuse of layer-0 value channels:

- `32.8%` of native stable edges joined the same head index across layers, versus a random-pair baseline of `10.6%`;
- removing the layer-0 value bus preserved only `44.8%` of native graph edges;
- the bus-robust intersection contains 39 edges and 21 nontrivial components.

Consequently, semantic claims below are restricted to bus-robust motifs. Same-index cross-layer recurrence by itself is treated as an architectural family, not evidence of a shared concept.

The native graph is not merely an address graph or a write graph. Address-only edges overlap the native graph by `8.3%`; write-only edges overlap it by `60.7%`. Weight discovery needs the pair, even though held-out semantic selectivity for the two clearest motifs turns out to live mostly in addressing.

## Accepted interpretation 1: prior-occurrence routing

### Frozen native heads

- `L2H5`
- `L3H5`
- `L3H8`

This three-head component survives removal of the shared value bus.

### Concrete discovery examples

Its strongest natural edges repeatedly connect the same surface form:

- `Kathy → Kathy`
- `PACK → PACK`
- `O → O`
- `APA → APA`
- `Mae → Mae`
- `schools → schools`
- `change → change`

One edge, `us → we`, suggested a broader reference-continuity offshoot. That motivated, but did not establish, a recursive-generalization interpretation.

### Held-out evidence

On untouched FineWeb rows 19,200–19,999:

- joint route-energy enrichment on prior exact-token matches: `10.22×`;
- same-layer all-head median: `1.78×`;
- QK address enrichment: `11.12×`;
- OV source-write-norm enrichment: `0.84×`.

All three frozen heads show the address effect (`11.36×`, `5.66×`, and `11.12×`). Held-out causal examples include:

- `Dean → Dean`, immediately before “Dean just…”;
- `Rodriguez → Rodriguez`, immediately before “Rodriguez pitched…”;
- `Toledo → Toledo`, immediately before “Toledo goes…”;
- `Roth → Roth`, immediately before “Roth IRA…”.

Removing the full semantic edge has much larger downstream effects than removing the closest nonmatching edge, but that comparison is magnitude-confounded: the available controls are typically only `20.7%` as large. When the semantic edge is only partially removed to equal the control's exact local-write magnitude, the median paired KL ratio falls to `1.07` with `11/18` wins. Thus the special mechanism is **where the head routes**, not an unusually potent write direction.

### Recursive-generalization verdict

- **Validated trunk:** exact surface-form recurrence.
- **Plausible offshoot:** entity/reference continuity across changed forms.
- **Not yet validated:** the offshoot. The `us → we` example is insufficient by itself.

Calling the whole motif a coreference circuit would currently overstate the evidence. A frozen coreference panel with names, pronouns, aliases, abbreviations, and adversarial repeated-but-unrelated strings is the next discriminating test.

## Accepted interpretation 2: immediate-left composition

Two distinct weight motifs independently implement a closely related routing verb.

### Motif A

- `L1H3`
- `L2H8`
- `L3H7`

Examples include:

- `2 → /` in a fraction;
- `mid → -` and `post → -`;
- `nature → of`, `rest → of`, `terms → of`;
- `way → to` and `exceptions → to`.

### Motif B

- `L1H5`
- `L2H6`

Examples include adjacent name/title and grammatical pairs such as `Sir → Ronald`, `remains → a`, and `you → get`.

### Held-out evidence

For motif A:

- joint distance-one enrichment: `21.82×`;
- QK address enrichment: `20.44×`;
- OV write enrichment: `1.00×`;
- same-layer address control median: `7.41×`.

For motif B:

- joint distance-one enrichment: `20.66×`;
- QK address enrichment: `20.06×`;
- OV write enrichment: `1.00×`;
- same-layer address control median: `7.62×`.

Held-out examples include `sake → of`, `Effects → Of`, `twist → of`, `non → -`, `pages → 151`, and `been → in`.

After exact local-write magnitude equalization, motif A is at parity with matched controls (paired KL ratio `1.03`, `9/18` wins). Motif B remains somewhat more behaviorally effective (paired ratio `1.88`, `8/12` wins), but the sample is small.

### Interpretation

“Read the immediately preceding token” is a clean reusable verb. It does not imply that all five heads have the same object-level meaning. The two weight motifs can route the same variable and apply different downstream transforms. This is a concrete example of why a clean computational role need not be a one-word concept.

The apparent variety—fractions, hyphens, names, and prepositional phrases—is not good evidence of arbitrary polysemanticity. Those are ordinary uses of the same local composition operation. This is the strongest finished-checkpoint analogue of recursive generalization in the current experiment, although temporal branch growth remains untested.

## Rejected interpretation: one multiscale boundary family

The frozen motif contained `L5H0`, `L5H7`, `L6H1`, `L9H8`, and `L10H5`. Discovery examples included end-of-document self edges, document-start reads, punctuation self edges, and longer section transitions. It was tempting to describe these as branches of one segmentation abstraction.

Held-out results do not justify that merge:

- median joint boundary enrichment: `1.36×`;
- QK address enrichment: `1.42×` versus `1.13×` layer controls;
- per-head address enrichments range from `0.74×` to `2.86×`;
- magnitude-equalized causal KL ratio: `0.88`, with `12/30` wins.

The correct conclusion is that **some member heads are boundary-sensitive**, especially `L9H8`, but the weight motif is not one validated semantic family. Recursive generalization cannot be used as permission to merge any locally related examples. This component should be split and retested.

## Other frozen motifs

Several motifs remain deliberately unlabeled:

- A two-head `L9H6/L12H4` motif has an intriguing weight decode contrasting singular `it/its` with plural `they/their`, but its natural routes do not yet establish a reference-number operation.
- A same-head-index motif across `L7H3/L10H3/L15H3` often appears around numbers, tables, and list-like text. It survives the bus-removal intersection but remains vulnerable to a generic depth-continuity explanation.
- Several larger components mix strong local syntax with long-range entity and document routes. Their current context panels do not support a hard-to-vary common description.

“Unexplained” is the correct label for these—not “polysemantic,” and not “recursively generalized.”

## What happened to the “find 50 circuits” experiment

It did not find 50 circuits. It began with 50 predeclared token properties; 32 had enough support and were capabilities the model could predict. Those 32 are a useful silver-standard ontology, not ground truth about internal variables.

Three results matter:

1. **Supervised activation readout works, but is distributed.** At Layer 4, the median held-out AUROC rises from `0.607` for the best native MLP channel to `0.779` for a frozen 64-channel coalition. No feature had a single channel within `0.03` AUROC of its 64-channel coalition. After exact current-token matching, the 64-channel median remains `0.622`, so embedding reinjection does not explain all of the signal.
2. **Those fixed neuron coalitions are not reusable causal circuits.** Across Layers 0–3, frozen 64-channel patches move the intended feature direction on only `31–61%` of pairs and have near-zero or negative mean semantic effects. Endpoint-informed top-64 controls work on `100%` of pairs. The intervention pipeline is capable; the fixed activation-defined coalition is the wrong causal object.
3. **A shared operator scaffold is real.** Across all 32 features, the next-eight-layer QK routing program recovers median exact-gradient cosine `0.901–0.969`, is directionally correct on `100%` of the held-out edits, and retains `95.2–98.4%` of the corresponding full near-window effect. This is evidence for reusable routing verbs, not 32 feature-specific head circuits.

The new frozen-cluster comparison used 1,600 sites: 50 uniformly sampled positive FineWeb contexts for each competent feature. Its head gates reproduce native logits exactly at all-one settings. A 10% finite downscale preserves the predicted sign on all 12 prospective checks, but magnitude correlation is only `0.860`, below the locked `0.95` gate.

The main signed comparison does not validate semantic cluster recovery:

- semantic split-geometry replication: `r = 0.507`, versus `0.182` for the Gram-matched nonsense ontology;
- all within-component signed cosine enrichment: `+0.041`, `p = 0.080`;
- direct frozen-edge enrichment: `+0.033`, `p = 0.197`;
- semantic-minus-null enrichment: `+0.022`, below the locked `+0.030` gate;
- each component has broad effective support over roughly `19–30` of the 32 labels rather than a clean one-label identity.

Unsigned importance is much easier to recover. Absolute centered profiles give semantic closure/direct-edge enrichments of `+0.115/+0.177`, but the Gram-matched nonsense ontology gives even larger `+0.130/+0.206`. Thus independently weight-similar heads tend to be important on similar examples, but the present test cannot say that they share the human semantic feature.

The correct use of the supervised set is therefore: freeze a mechanism first, attach a multi-label anchor profile afterward, and require semantic-over-null evidence before treating a label as an interpretation. Folding the labels into clustering would turn a useful external check into circular evidence.

## What this says about representation

The evidence favors the following provisional ontology:

1. **Weights define available verbs.** The symmetry-invariant weight geometry finds locally recurring address/write operators.
2. **Inputs invoke those verbs.** For the clearest motifs, semantic organization appears as selective routing on natural data.
3. **Ordinary write directions carry the result.** Recurrence and local composition do not use specially potent OV directions per unit local-write magnitude.
4. **Dense readers remain.** Downstream MLP reader weights are broad; the earlier hypothesis that each motif would terminate in a small reader set failed.
5. **Local motifs are more natural than global types.** A forced exhaustive clustering creates unstable stories. Stable nearest-neighbor structure is real but incomplete.

This does not prove that the model is globally monosemantic or free of superposition. It does show that examples which look heterogeneous at the object level can be clean at the operation level. “Names, acronyms, institutions, and repeated words” can all be instances of **retrieve a previous occurrence**; “fractions, hyphens, names, and prepositions” can all be instances of **compose with the immediate left token**.

## Standard for recursive generalization going forward

An apparent branch should be accepted only if it passes progressively stronger gates:

1. **Local bridge:** a specific relation connects the branch to the trunk.
2. **Frozen prediction:** the proposed parent predicts untouched examples, including counterexamples.
3. **Mechanistic agreement:** related examples use the same weight motif and routing/read pathway.
4. **Causal relevance:** the branch's native invocation affects downstream behavior.
5. **Temporal growth:** where checkpoints exist, the branch grows from the predicted local origin in the predicted order.

This model currently allows gates 1–4. Gate 5 requires training checkpoints or a controlled retraining run.

## Next experiments

1. **Recurrence-to-reference ladder:** predeclare 50+ examples each of exact repetition, capitalization changes, aliases, acronym expansion, singular/plural pronouns, and unrelated lexical repeats. This directly tests whether the proposed offshoot is a predictable branch.
2. **Split the boundary component:** discover submotifs separately for document reset, sentence boundary, line/list boundary, and long-range section routing; do not fit one umbrella label first.
3. **Trace the two composition motifs into MLP readers:** compare their weight-distinct OV maps and downstream neuron effects to determine what each does with the same previous-token variable.
4. **Training-time replication:** save dense early checkpoints in a smaller matched model and ask whether recurrence grows from exact copying toward reference continuity, rather than merely coexisting at the final checkpoint.
5. **Viewer integration:** expose frozen motif membership, bus robustness, held-out address enrichment, accepted/tentative labels, counterexamples, and finite causal results. Do not display a semantic name without its evidence grade.
6. **Upgrade supervised anchors from token properties to controlled computations:** use 50+ examples each for recurrence, delimiter matching, agreement, copying, induction, and other tasks the model demonstrably performs. Token-class labels mostly test what output is likely; controlled minimal pairs can test what operation was used.

## Artifacts

- `bilin18_unsupervised_weight_circuit_atlas.json`: frozen discovery and natural examples.
- `bilin18_unsupervised_weight_motif_redteam.json`: bus/address/write architectural controls.
- `bilin18_unsupervised_motif_semantic_validation.json`: held-out route and full-edge causal results.
- `bilin18_unsupervised_motif_semantic_causal_redteam.json`: exact magnitude-equalized interventions.
- `bilin18_unsupervised_motif_route_decomposition.json`: QK address versus OV write enrichment.
- `bilin18_supervised_anchor_motif_comparison.json`: 50-context-per-feature causal fingerprints versus frozen motifs.
- `bilin18_supervised_anchor_metric_redteam.json`: signed/magnitude and semantic/Gram-matched metric controls.
