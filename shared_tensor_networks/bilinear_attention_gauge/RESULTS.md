# Bilinear-attention gauge canonicalization: completed small-model comparison

Current research design: [reduction metric × legibility-prior matrix](CANONICALIZATION_RESEARCH_MATRIX.md).
Latest toy graph prior: [sparse key directions plus reusable quadratic nodes](THREE_COUNT_GRAPH_PRIOR_RESULTS.md). Moderate graph cost improves held-out node-pruning fidelity while preserving most designed-variable alignment in ten scratch-trained attention models; independent mechanism transfer and physical circuit recovery remain open.
Independent-seed red team: [ten fresh trained models](THREE_COUNT_GRAPH_PRIOR_FRESH_RESULTS.md) weaken that result. Uniform-cut pruning improves at a larger variable-alignment cost, while the frequency-cut gain nearly vanishes with fixed hyperparameters. The next prior should optimize simplification error directly rather than a saturated count of nonzero quadratic coefficients.
Latest red team: [pair-level labels versus full-model directional key edits](ALL_HEAD_DIRECTIONAL_REVERSAL_RESULTS.md). The labelled cells separate in an isolated pair table, but the proposed full-context double dissociation reverses on 48 new windows. The next legibility target should use context-level edit responses or named task outcomes.
Nonlinear toy follow-up: [rank-two metric tradeoff across ten trained attention checkpoints](THREE_COUNT_NONLINEAR_METRIC_RESULTS.md). Preserving an analytic bilinear readout raises full-test parity accuracy to .931 but loses a withheld rare combination entirely; a rank-three weight-only cut preserves both.
Fixed-function legibility follow-up: [key sparsity versus context-edit sparsity](THREE_COUNT_NONLINEAR_GAUGE_RESULTS.md). Key sparsity recovers the designed A/B key directions (mean cosine .996) but exposes a real cross-axis parity interaction. Context-edit sparsity mixes those directions while making single-axis deletion effects more additive. This separates intervention sparsity from variable recovery.
Trained composition control: [learned bilinear readouts on ten count-attention models](THREE_COUNT_TRAINED_COMPOSITION_RESULTS.md). The same exact-gauge conflict survives both frozen-backbone training and joint fine-tuning: key L1 aligns variables, while edit-sparsity mixes them and hides the product interaction in single-axis effects.
From-scratch control: [ten jointly trained two-stage attention models](THREE_COUNT_SCRATCH_COMPOSITION_RESULTS.md) all solve the rare holdout. Key L1 again recovers variable directions more accurately than an edit-sparsity gauge; a weight-only quadratic diagonalization recovers much of the latter's edit sparsity while mixing the variables.

Rank-96 fidelity and causal panel: [full-vocabulary, natural full-model, and QK2-intervention comparisons](RANK96_METRIC_AND_CAUSAL_PANEL.md). A uniform weight-only QK1 cut best preserves natural next-token distributions across three corpora; a FineWeb-weighted cut better survives a strong QK2-constant edit. This is a measured objective tradeoff before the next legibility-prior sweep.

Rank-96 legibility-prior panel: [exact sparse gauges, held-out label transfer, and calibrated full-model ablations](RANK96_LEGIBILITY_PRIORS.md). Activation sparsity lowers effective active-coordinate counts on external text for both fidelity priors, but axes vary across seeds, coarse token-form labels are weaker than simple baselines, and causal localization does not transfer consistently. The next test needs grouped functional or downstream-sensitive targets.

Group-stability follow-up: [energy-selected rank-96 groups across seeds and random starts](RANK96_GROUP_STABILITY_RESULTS.md). Four-axis sparse groups have strong projector and pair-response agreement despite weak single-axis alignment, and their calibrated full-model document-effect profiles are more reproducible than random groups on three corpora. This establishes one stable high-energy functional subspace within a chosen bond, not semantic or multi-group discovery.

Multi-group red team: [disjoint energy-ranked quartets and a null-calibrated co-activation graph](MULTIPLE_RANK96_GROUPS_RESULTS.md). The first quartet remains robust; a second has partial pair and causal-profile transfer, while third/later quartets and a simple graph do not yield a reliable multi-group explanation. The next target is a functional or downstream-causal grouping objective with task-level holdouts.

Trained three-rule toy follow-up: [intervention-aware six-dimensional key space and unsupervised multi-group recovery](THREE_COUNT_CAUSAL_GROUPING_RESULTS.md). Sparse axes grouped by their ordinary-function query/output signatures recover complete A/B/C variable groups in 10/10 trained seeds and give selective held-out task deletions across three count rates. Naive stacked-intervention cosine and sentence co-activity fail; fitting single branch edits does not faithfully predict their withheld double edit. This separates causal subspace coverage from semantic grouping.

Real-model position follow-up: [RoPE transport and held-out functional grouping on the 546M head](FLAGSHIP_ROPE_GAUGE_TRANSPORT_RESULTS.md). A larger sampled-token panel finds only small null-calibrated groups. Fixed post-RoPE component responses change drastically under a global position translation, even as the summed response changes little; co-rotating the component basis restores the response but changes the cut. A position-covariant object is needed before treating these groups as an ontology.

Pre-RoPE relocation control: [same-rank fixed versus position-covariant cuts on held-out tokens and positions](PREROPE_CUT_TRANSFER_RESULTS.md). Moving either existing rank-96 cut before RoPE preserves its reference function and same-gap translation behavior, but raises new-gap complete-head response error substantially. Position covariance alone does not buy fidelity; a future matched multi-offset fit must establish whether a useful pre-RoPE cut exists.

Matched multi-offset result: [separately fitted pre- and post-RoPE rank-96 cuts](MATCHED_PREROPE_FIT_RESULTS.md). Under the same uniform/corpus pair objective and WTG procedure, the pre-RoPE cut has lower held-out pair error, survives a QK2-constant pair edit better, and cuts full-model KL by about 3.3–4.4× on three corpora. Specific held-out position-gap panels still favor post-RoPE. This reverses the naive relocation outcome and makes position-covariant legibility priors worth testing, subject to fidelity gates.

Metric factorial: [uniform-only, corpus-only, and mixed reductions in both RoPE geometries](ROPE_METRIC_FACTORIAL_RESULTS.md). The mixed pre-RoPE cut has the lowest full-model KL on FineWeb, WikiText-103, and TinyStories, including a smaller two-seed replication. Uniform pre best protects uniform pair fidelity; corpus pre best protects a corpus QK2-constant pair edit. This chooses a fidelity baseline for the next exact-gauge legibility-prior sweep, not a semantic interpretation.

Pre-RoPE legibility follow-up: [exact sparse gauges, functional groups, held-out labels, and calibrated full-model deletion](PREROPE_GAUGE_AND_GROUP_RESULTS.md). Exact rotations of the fixed mixed cut lower effective active-coordinate counts on held-out pairs. A factor-weight gauge yields a moderately stable high-energy response subspace across random starts. Its connective-key enrichment transfers to three corpora, but frequency-matched token controls expose uneven specificity; full-model group deletion is evaluated separately.

Conditional causal follow-up: [key-conditioned group deletion with matched-token, nearby-position, and new-document controls](PREROPE_CONDITIONAL_LEXICAL_RESULTS.md). A broad post-hoc connective label collapses mainly to the tokens `" as"` and `" so"`. On new documents across three corpora, deleting the group only at those keys causes larger full-model KL than either control family, but a few windows dominate and the group has much larger raw head response there. This supports a narrow lexical causal association within the fixed reduced circuit, not a syntax ontology.

Group-and-dose red team: [independent-start versus random subspaces, smaller deletions, and edited-prediction examples](PREROPE_GROUP_DOSE_AND_PREDICTION_RESULTS.md). Independent-start groups reproduce the `" as"`/`" so"` effect at 25–100% deletion, but random subspaces can cause similar KL at comparable removed head energy. The robust result is selective group activity on these keys, not privileged downstream sensitivity or an established grammar circuit.

All-head causal follow-up: [full-model response to punctuation-key versus word-key input edits](ALL_HEAD_COUNTERFACTUAL_CAUSALITY_RESULTS.md). A validation-selected response-varimax component changes the punctuation-edit logit response more than the word-control response on 96 new windows across three corpora and two punctuation sets, compared with energy-matched random and same-gauge controls. The rank-256 baseline preserves the native model's input-edit response closely. The effect is small and belongs to the chosen reduced explanation; named-task and physical-node claims remain open.

All-head legibility red team: [post-fit token-form labels on new documents and corpora](ALL_HEAD_LABEL_TRANSFER_RESULTS.md). Direct token-response L1 concentrates components, but response varimax transfers coarse query/key form labels more strongly on the larger clipped-energy panel. Sparse activity and label enrichment rank the same exact gauges differently.

Exact-gauge all-head follow-up: [compare weight and token-response sparsity priors at fixed rank 256](ALL_HEAD_GAUGE_PRIOR_RESULTS.md). Direct token-response L1 from the supplied basis improves held-out token and pair-component concentration without changing the function. Weight sparsity makes axes more head-local but worsens response concentration; random-start L1 remains nonunique. Semantic and selective causal validation is still required.

Function-aware all-head follow-up: [fit a rank-256 shared key subspace to the summed nine-head response](FUNCTION_AWARE_SHARED_KEY_RESULTS.md). A larger mixed uniform/corpus fit improves natural pair fidelity and held-out full-model KL against the same-rank usage-SVD basis, but its advantage largely disappears when QK2 is held constant; intervention fidelity remains a separate objective.

All-head shared key result: [weight-only versus usage-weighted dictionaries across all nine first-layer heads](ALL_HEAD_SHARED_KEY_RESULTS.md). A shared rank-256 token-key dictionary uses 45.3% of the factor entries of nine separate rank-64 dictionaries while matching or beating their held-out natural pair response. Full-model propagation is mixed at rank 256; a shared usage-weighted rank-384 dictionary uses 68.0% of the entries and beats the separate control on 56/64 FineWeb, 59/64 WikiText, and 63/64 TinyStories paired windows across two disjoint document panels. This is native-key-table compression and KL fidelity, not a gauge-invariant canonical function or semantic feature recovery.

Follow-up: [metric × legibility-prior plan and known-rule two-head toy results](FACTORIAL_PLAN_AND_TOY_RESULTS.md). This follow-up separates uniform versus usage-weighted reduction from native-bond, WTG-bond, pair-response, and intervention objectives. It is a hand-built toy and does not establish interpretation of a trained model.

Flagship metric follow-up: [fixed uniform–corpus and intervention mixtures on the 546M head](MIXED_METRIC_RESULTS.md). A half uniform–corpus fit improves held-out sequence head-output fidelity on FineWeb, WikiText-103, and TinyStories at matched rank; adding a pair-level QK2 intervention objective does not improve sequence-level interventions. Sparse and human-legibility priors remain untested on this head.

Sequence-objective follow-up: [directly optimize the summed first-layer head response](SEQUENCE_OBJECTIVE_RESULTS.md). With disjoint FineWeb fit/selection/test documents and two separate test corpora, rank-64 sequence fits beat the initial WTG cut and a direct pair-refit control. A 25% sequence-level QK2-constant objective improves that intervention and a held-out QK2 key shift. Full-model behavior and component legibility remain open.

Full-model follow-up: [propagate the saved cuts through all 18 layers](FULL_MODEL_BEHAVIOR_RESULTS.md). On 18,432 next-token positions across disjoint FineWeb, WikiText-103, and TinyStories samples, the natural sequence fit has the lowest original-to-edited KL in both seeds of each corpus. A zero-head edit has roughly 47–54 times its KL. CE shifts are tiny and do not establish an accuracy gain; component legibility remains open.

Exact-gauge legibility-prior follow-up: [factor-weight versus activation sparsity on the fixed rank-64 cut](EXACT_GAUGE_SPARSITY_RESULTS.md). Orthogonal rotations preserve the entire attention function to numerical precision. Activation objectives lower effective active-component count on held-out corpora, but the gain concentrates on common tokens and independently fitted component bases align weakly. Factor-weight L1 drops sharply without comparable pair-response sparsity. These are sparsity proxies, not semantic components.

Label red team: [post-fit token-form labels and exemplars for the exact-gauge components](COMPONENT_LABEL_TRANSFER_RESULTS.md). At 262,144 pairs per corpus, activation-sparse gauges do not beat random or SVD gauges on coarse held-out label enrichment. The label taxonomy is dominated by broad word-form classes; the subsequent full-model ablation tests context localization separately.

Component-causality red team: [full-model deletion and matched partial deletion of exact-gauge components](COMPONENT_ABLATION_RESULTS.md). Top activation-sparse components have larger uncalibrated effects because they carry more head response. After calibrating all edits to the same head-output error on separate documents, their downstream KL is no more localized across contexts than random-basis components in this limited assay.

Trained follow-up: [two-head toy-language reduction and red team](TRAINED_TOY_LANGUAGE_RESULTS.md). The pair-function extraction is exact; whole-sequence and pair-frequency environments differ beyond sampling noise, while the task is too redundant for accuracy to distinguish the reduced heads.

Stronger trained follow-up: [necessary two-head dual-output language](DUAL_OUTPUT_LANGUAGE_RESULTS.md). Ten seeds establish a real two-head dependency; rank-one reduction usually preserves the task, while a difficult seed separates uniform, data-aware, task, and intervention criteria.

Joint function follow-up: [exact combined-table reduction, metric comparison, and sparse key dictionary](JOINT_FUNCTION_TOY_RESULTS.md). On ten trained dual-output toy checkpoints, a rank-two reduction of the summed token function preserves the rare withheld rule in every seed, including the difficult seed that fails a same-raw-factor-count headwise reduction. Uniform, training-frequency, and half-mixture metrics all preserve the task at rank two; none reliably rescues a rank-one bottleneck. A label-free sparse rotation of the rank-two function's key dictionary aligns with the two task bits and gives selective post-fit bit-flip effects. This is an interpretable **function-level** basis, not recovery of physical head computations.

Physical-head red team: [project the joint dictionary onto each trained head and test branch interventions](JOINT_FUNCTION_PHYSICAL_REDTEAM.md). The fixed two-direction key space reconstructs the original physical head tables with mean .00469 relative MSE, far better than random key spaces. Both heads mix A and B. The same dictionary fails held-out QK1- and QK2-constant interventions (.465 and .574 mean relative MSE), although each edited table is exactly rank two. The semantic function basis is therefore not a branch-robust physical circuit explanation.

Intervention-aware follow-up: [fit a shared key space across ordinary and branch-edited functions](INTERVENTION_DICTIONARY_RESULTS.md). A rank-two ordinary-function space fails the two branch edits; including one edit at rank two fails to transfer to the other. Fitting the original and both single edits at rank four yields under .012 mean relative table MSE for each, yet misses a held-out **double edit** at .105 error. Including that combination and using rank five reduces its error to .00406. This measures the extra representation budget needed for specified internal counterfactuals, without yet identifying a physical circuit graph.

Sparse-basis follow-up: [rotate the intervention-aware key spaces without task labels](INTERVENTION_SPARSE_BASIS_RESULTS.md). Adding branch-edit coverage changes the representation from one sparse axis per bit to roughly two token-specific axes per bit. At rank four, mean single-axis A/B cosine is only about .72, but the full subspace covers the A/B contrasts at .986–.988 and sparse components have .978–.981 post-fit bit-flip effect purity, well above SVD and random gauges. Axis alignment alone would misread this as semantic failure; purity alone would overclaim unsupervised grouping.

Unlabeled-grouping follow-up: [pair sparse atoms by sentence co-occurrence](UNSUPERVISED_GROUPING_RESULTS.md). With a supplied two-pairs-of-two structure and only 32 unlabeled sentences, the rank-four sparse axes form groups with selective A/B effects in 10/10 seeds; SVD axes do so in 4/10 or 2/10, and random rotations around 30%. Group deletion preserves the other bit's exhaustive accuracy. This is constrained group recovery, not discovery of an unknown number of concepts or physical nodes.

Variable-size grouping follow-up: [correlation-graph groups on a five-direction intervention-aware basis](GRAPH_GROUPING_RESULTS.md). Without supplying group count or sizes, 32 unlabeled sentences identify *selective-effect* A/B groups in 10/10 seeds, versus 3/10 for SVD and about 10% for random rotations. The stronger whole-contrast criterion succeeds in 9/10 sparse seeds; a third group is filler-specific in nine. The toy grammar makes mutual exclusion highly informative.

Grammar-shift red team: [vary unlabeled key counts while holding the trained sparse dictionary fixed](GROUPING_GRAMMAR_SHIFT_RESULTS.md). The complete-variable criterion collapses from 9/10 under exactly one token per family to 0/10 at the original .8 threshold with family dropout, variable counts, or independent Poisson token counts. The earlier effect-only criterion falsely remains 10/10 because singleton axes are individually selective. This identifies a concrete failure of co-occurrence-only grouping and motivates adding functional or intervention evidence.

Functional-grouping follow-up: [connect sparse atoms by exact query/output response similarity](FUNCTIONAL_GROUPING_RESULTS.md). Without sampled sentences, the weight-derived original-function signatures recover complete A/B groups in 9/10 seeds at thresholds .8–.9, versus 2/10 for SVD and roughly 10–21% for random gauges. This bypasses the Poisson co-occurrence failure on the same toy, while leaving one difficult seed and real-language generalization open.

Variable-count trained-language follow-up: [a new two-head count-comparison task](COUNT_LANGUAGE_CANONICAL_RESULTS.md). Ten models train on 50,000 variable-count sentences each; nine fully solve the rare withheld query/bit combination, while one reaches 98.66%. The exact combined function is nearly rank two and a weight-only rank-two reduction retains mean .999985 full-test and .99960 rare-heldout joint accuracy. Training-frequency weighting improves independent test-logit fidelity at the cost of uniform table fidelity; all rank-one metrics fail. Label-free key sparsity aligns its two axes with A/B count contrasts at mean cosine .99927 and gives selective function-component ablations across three count rates. This is task-grounded semantic function recovery, not physical circuit recovery.

Variable-count intervention follow-up: [test the semantic basis against physical QK-branch edits](COUNT_INTERVENTION_RESULTS.md). The ordinary two-direction key space is badly unfaithful to single and double edits. A four-direction space fitted across the original and both single edits reaches below .006 mean error on those conditions but leaves .0588 on the held-out double edit; fitting all four cuts each mean error below .010. Causal intervention coverage and ordinary semantic simplicity remain distinct objectives.

Three-variable rank-budget follow-up: [train a three-head count language and compare reductions](THREE_COUNT_CANONICAL_RESULTS.md). Ten models solve all three count-sign bits and a rare whole-combination holdout. Their exact combined token table has three large singular directions: joint rank three preserves 100% task accuracy while rank two falls to .571 mean full-test accuracy. Under rank two, uniform, training-frequency, and supervised CE objectives trade ordinary accuracy, rare-heldout behavior, and teacher-logit fidelity differently. At rank three, label-free key L1 aligns all three axes with the A/B/C rules at mean cosine .99950 and gives selective explanatory-component deletions. Physical QK interventions and real-language transfer remain separate tests.

Nonlinear follow-up: [two-head count language plus bilinear MLP for XOR](XOR_COUNT_CANONICAL_RESULTS.md). A weight-only rank-two cut nearly preserves ten trained models, while rank one fails. Pair-frequency and sequence-aware reductions change fidelity slightly, but not accuracy at rank two. Among exact-gauge legibility priors, key-dictionary L1 uniquely recovers near-pure A/B axes; deleting either disrupts its bit and XOR while largely preserving the other bit. The original downstream MLP's signed cross monomial alone predicts XOR on an independent test. The rank-two semantic dictionary fails QK-branch interventions; fitting original and single edits needs four or five dimensions and still incompletely predicts the double edit. One seed fails the rare A/B ordering despite strong validation, and its failure remains in all summaries.

Flagship key-prior transfer: [test key-side L1 on the fixed rank-64 546M-model head cut](FLAGSHIP_KEY_SPARSE_RESULTS.md). Uniform-domain and FineWeb key sparsity change component gauges without changing the attention function. The FineWeb prior lowers key L1 and yields more stable components across fit seeds, but neither prior improves coarse held-out token-form labels. Calibrated full-model deletions do not establish a general localization advantage over SVD. Post-hoc inspection finds a strong FineWeb “as much” bigram effect for one component, with much weaker and undersampled external-corpus transfer; it remains an exploratory lead rather than a canonical semantic circuit.

Full-function rank follow-up: [compare flat complete head response, branch-swap-invariant effective score, and structured QK1 bond cuts](FULL_FUNCTION_RANK_AND_SCORE_RESULTS.md). Exact full-vocabulary-key SVD shows a flat head-response rank-64 cut leaves .683 uniform training error even at fixed position; rank 256 still leaves .476. Folding out the exact value/OV map improves the effective-score rank-64 fit substantially, but held-out query/position errors remain high. Training on six relative offsets helps without closing the gap. On matched natural adjacent pairs, data-aware score cuts improve over a flat function cut, while the structured bond often remains more faithful. This is evidence that preserving multiplicative reuse matters more than sparsifying a poor flat unfolding; no full-model KL or physical branch recovery is claimed for the new function cuts.

Legibility-prior follow-up: [key-family and query-conditional group penalties](DUAL_GROUP_PRIOR_RESULTS.md). The supplied token-family partition and a broader intervention panel do not rescue the hard held-out rule at rank one; group concentration is not itself a mechanistic success criterion.

Identifiability control: [shared versus duplicated attention computation](SHARED_COMPUTATION_IDENTIFIABILITY.md). Two graphs have exactly the same external function, including input-edit responses, while one internal-node intervention distinguishes their reuse structure. Shortest-description structure is a prior, not proof of the implemented graph.

Full-width flagship follow-up: [matrix-free first-layer heads](FULL_WIDTH_FLAGSHIP_RESULTS.md). The width-128 physical bond is reduced without an eight-token domain quotient. Uniform and corpus measures, a branch-swap control, and a matched direct-fit comparison separate gauge truncation from the metric it is meant to optimize.

Context-transfer follow-up: [length, rare-token, cross-corpus, and sequence-output tests](FLAGSHIP_CONTEXT_TRANSFER_RESULTS.md). The original four-position metric overweights self-pairs; length-128 fitting and larger ranks improve long-context head fidelity but still do not establish downstream CE or semantic legibility.

This experiment tests Glen Evenbly's [weighted trace gauge (WTG) and full-environment truncation (FET)](https://arxiv.org/abs/1801.05390) on a repeated-token, bilinear-attention circuit. It distinguishes three claims:

1. A fixed attention-network representation has a reproducible internal gauge.
2. Its bond can be compressed more faithfully using the full loop environment than by a local SVD.
3. The resulting gauge identifies a canonical decomposition of the **function**.

The first two claims hold in this controlled setting; the third fails under an exact QK-branch swap.

## Object and metric

Four existing trained, one-layer checkpoints from `tensor_language/runs` have head widths 8, 16, and 32, vocabulary size 100, and causal length 8. Three have one head; the fourth has two width-8 heads, both evaluated separately while the other head remains fixed in the full model. The original first-layer head is represented exactly on tokens by

\[
T_{ijabo}=\mathbf 1_{j\leq i}
\left(\sum_r U_{iar}V_{jbr}\right)B_{ijab}C_{bo},
\]

where \(U,V\) are the first rotary QK branch, \(B\) is the second QK score including the head scale, and \(C\) is the OV map. Positions \(i,j\) remain explicit: the stored rotary tables are rounded separately, so collapsing to an exact relative-position table would be numerically false. The full-rank token-table forward matches the saved model at max absolute residual error below \(1.3\times10^{-14}\).

The compression target is the internal QK1 bond: insert a rank-\(r\) matrix \(X\) in place of its identity. This is an after-RoPE bond operation. The expected squared error averages over valid causal token pairs under a specified token distribution. At \(i=j\), only equal query and key tokens are allowed. The experiments use uniform tokens and a Zipf-1 sensitivity check. A separate set of 128 random sequences measures the resulting first-layer logits. No sequence samples are used to choose the decomposition.

Methods:

- **Local SVD** first truncates QK1 while cutting away QK2 and OV.
- **WTG** forms the exact double-layer bond environment, solves the dominant eigenoperator problems, fixes the weighted trace gauge, and retains its largest bond coefficients.
- **Full-environment ALS** alternates exact least-squares updates for a rank-\(r\) bond matrix using that environment. Its objective is the same best-scaled state-fidelity objective as FET, although this is an ALS implementation rather than a line-by-line port of Evenbly's update. Convergence is local; it is not a global-optimum certificate.
- **Collapsed functional SVD** truncates a different unfolding, \((i,a)\) versus \((j,b,o)\), of the entire token function. Invalid self-pairs are zero-completed. Its rank and parameter count are **not** matched to an internal-bond rank; it is a descriptive functional baseline.

## Truncation outcomes

Relative squared token-function error under uniform valid-pair weighting:

| Checkpoint / head | Bond rank | Local SVD | WTG truncation | Full-environment ALS | Collapsed functional SVD |
|---:|---:|---:|---:|---:|---:|
| Width 8 / sole | 2 | 0.002064 | 0.000844 | **0.000366** | 0.186 |
| Width 16 / sole | 2 | 0.011753 | 0.003687 | **0.000505** | 0.133 |
| Width 32 / sole | 4 | 0.000259 | 0.000169 | **0.000073** | 0.142 |
| Width 8 / head 0 of 2 | 1 | 0.032234 | 0.022367 | **0.018958** | 0.306 |
| Width 8 / head 1 of 2 | 1 | 0.001120 | 0.001075 | **0.000325** | 0.368 |

At these tight ranks, the loop-aware truncations improve the exact function metric, and the full-environment solution is best. The same ordering holds for relative logit MSE on the sampled sequences: width 16/rank 2 gives 0.01153, 0.00365, and 0.000526 respectively. At higher ranks the differences narrow; all ranks, sequence outcomes, alternate metrics, and objective traces are in the JSON files. The collapsed SVD's large error reflects a different global unfolding (especially position and causal-mask structure), not a fair resource-matched defeat. Its valid-domain weighted error need not decrease monotonically because its SVD optimizes an unweighted, zero-completed tensor.

The Zipf-1 width-16/rank-2 run gives 0.011738, 0.003839, and 0.000424 under its own metric; this is a sensitivity check with an artificial token frequency law, not a language-model corpus result. Cross-metric scores for every solution are also saved.

## Gauge and interpretation checks

- An invertible, non-orthogonal QK1 bond gauge changes the raw factors while preserving the exact token function to below \(3.2\times10^{-14}\). WTG-truncated token functions agree across that transformation to below \(10^{-11}\) in all recorded ranks. Normalized WTG coefficients agree to below \(1.4\times10^{-14}\), and the normalized boundary matrices are identity to below \(1.3\times10^{-12}\). The bridge control has cycle entropy about zero, and WTG and full-environment truncation coincide there.
- **Exact branch-swap counterexample.** Swapping the two QK branches leaves the entire head token function identical to within \(7.2\times10^{-15}\). Yet cycle entropy on the now-targeted bond changes from **0.659 to 0.286 bits** (width 8), **0.959 to 0.383 bits** (width 16), and **0.673 to 0.593 bits** (width 32). In the two-head checkpoint, head 0 changes **0.258 to 0.303** and head 1 **0.336 to 0.640** bits. This is a representation change, not an internal gauge transformation of one fixed bond. It proves that these bond entropies are not invariants of the attention function and cannot alone identify which multiplicative branch is the physical or preferred mechanism.
- **Causal stress.** Setting QK2 to a constant while keeping QK1 and OV fixed generally preserves the tight-rank ordering here. For width 16/rank 2, intervention error is 0.01341 (local), 0.00363 (WTG), 0.000628 (full environment). Thus the in-distribution win does transfer to this one branch intervention; it does not establish broad causal fidelity. Every intervention score is retained in the JSON.

The small checkpoints do not test a joint decomposition that redistributes computation between heads. The earlier flagship gauge work also found that simple private-bond rotations bought little compression, so the small-model bond-truncation gain must not be read as a full-model gain.

## Bounded 546M-checkpoint pilot

`flagship_pilot.py` uses only cached weights from the 18-layer, nine-head bilinear model, without loading its full module or using the GPU. It examines first-layer heads 0 and 4 on two independently chosen sets of eight token IDs and four positions. The model's QK RMS normalization and bf16-rounded rotary tables are included. The raw post-RoPE QK1 bond has width 128, but on each selected domain its exact score matrix has rank 32. An SVD refactors that *restricted function* into a 32-dimensional bond before WTG. It is a domain quotient, **not** a gauge transformation of the full 128-dimensional physical bond.

An independent sequence calculation of all nine heads, including OV mixing, agrees with the sum of the token tables to max absolute error \(5.7\times10^{-14}\). The restricted QK quotient and QK-branch swap also pass exact-function checks. The checkpoint and chosen token IDs are recorded in `flagship_results.json`.

Uniform valid-pair relative squared error on the selected domain:

| Token seed | Head | Bond rank | Local SVD | WTG truncation | Full-environment ALS |
|---:|---:|---:|---:|---:|---:|
| 17 | 0 | 4 | 0.06007 | 0.06812 | **0.01014** |
| 17 | 4 | 4 | 0.00384 | 0.00288 | **0.00136** |
| 23 | 0 | 4 | 0.03685 | 0.06323 | **0.00408** |
| 23 | 4 | 4 | 0.02206 | 0.00766 | **0.00319** |

Here full-environment optimization wins all four rank-4 cases, but WTG truncation is **worse than local SVD in both head-0 cases**. The WTG is still gauge-consistent; canonical coefficients alone need not give the best truncation when cycle entropy is nonzero. At rank 8, full-environment errors are 0.000130, 0.0000288, 0.0000235, and 0.0000461 in the same case order. The seed-23/head-0 rank-8 ALS ran its 600-step cap and was still improving slowly, so that number is an upper bound on the best achievable error, not a converged optimum.

The exact QK swap again preserves the selected-domain function while changing cycle entropy: seed 17/head 4 gives **0.612 versus 1.593 bits**. This extends the representation-dependence finding to the flagship weights, but only on the declared token subset. No conclusion follows about full-vocabulary compression, later layers, CE loss, or whether a recovered bond names the model's physical computation. Exact WTG here uses a dense \(32^2\times32^2\) transfer eigensolve; applying it directly to a full width-128 head would require a matrix-free implementation and a tractable full-domain environment.

## Reproduce

Use a Python environment with PyTorch, NumPy, SciPy, and einops. For example, the local environment used here is `/home/loganriggs/Coding/neuron_circuits/.venv/bin/python`.

```bash
OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/neuron_circuits/.venv/bin/python run.py \
  --checkpoint /home/loganriggs/Coding/reverse_engineer/tensor_language/runs/L1_d16_h1/model.pt \
  --ranks 2 4 8 12 --output results_d16.json
```

The script asserts exact-forward matching, gauge invariance, WTG boundary conditions, agreement between literal and environment errors, and a bridge control. JSON contains all metrics and optimization traces; companion NPZ files contain the fitted bond matrices and WTG gauge matrices. The original model checkpoints are referenced by path and SHA-256 rather than copied into this repository.

The CPU flagship pilot runs with `OPENBLAS_NUM_THREADS=1` and `flagship_pilot.py`; its output is `flagship_results.json`.

## Decision

Evenbly's methods are useful for **bond-level compression within a chosen copy-tensor network**, especially when the bond is genuinely cyclic. They do not solve function-level canonicalization. The branch-swap counterexample is an exact, architecture-native reason to keep those objectives separate. The selected-domain flagship pilot supports testing a matrix-free environment on larger token coverage, but the next mechanistic experiment should compare candidate function-equivalent factorizations on held-out, branch-specific interventions; merely collecting more WTG spectra would not answer which computation the model uses.
