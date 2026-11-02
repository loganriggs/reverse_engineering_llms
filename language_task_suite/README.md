# Grounded toy-language task suite (first pass)

**Current follow-up:** [`WORLD_RESULTS.md`](WORLD_RESULTS.md) documents the newer latent document-world generator, its 32,000-example oracle audit, 1–4-block and larger-model results, causal-edit tests, final attention-bond gauge experiments, and the measured gap to FineWeb surface statistics. The earlier 12-family suite described below remains a simpler baseline. The document-world v1 has a known indefinite-article error (`a orange`); retain its source with its checkpoints and correct it in a versioned successor.

[`V2_RESULTS.md`](V2_RESULTS.md) follows that versioned successor. It lengthens and repairs the document realizations while preserving the latent-world task, checks oracle/edit/split invariants, and retrains the 1–4-block depth sweep. Select the generator explicitly with `--world-version v1` or `--world-version v2` in `train_world.py`, `validate_world.py`, `eval_world_counterfactual.py`, and `world_bond_gauges.py`.

**Research-direction correction:** [`EXISTING_GENERATOR_BRIDGE.md`](EXISTING_GENERATOR_BRIDGE.md) locates the earlier autoregressive mixed-rule generator and trained tensor-transformers in `toy_models_of_tensor_networks/attn_circuits`. Its competing n-gram, induction, and bracket distributions are the primary sandbox for the intended mechanism-discovery loop. The document-world benchmark is a complementary state-tracking stress test.

The [mixed-rule weight sweep](MIXED_RULE_WEIGHT_SWEEP.md) now changes bigram versus induction weights, retrains matched bilinear models, scores exact-oracle KL across mixtures, and tests fixed-prefix rule-competition shifts and attention-bond compression.

The [conditional mixture experiment](CONDITIONAL_MIXTURE.md) trains one model with an explicit rule-mixture token. It tests a causal tag intervention on fixed prefixes and finds a compact first-layer broadcast contrast that transfers from no-conflict fitting contexts to held-out bigram–induction conflicts, with an exact rank bound and positional controls.

The [weight-only tag-contrast analysis](WEIGHT_ONLY_TAG_CONTRAST.md) derives that first-layer broadcast directly from embeddings, QK products, and tag values over all token-position pairs. A four-direction data-free basis nearly matches contrast PCA on two disjoint conflict panels for both bigram-heavy and induction-heavy tag changes. This is intervention-specified local mediation, not task-blind discovery or full circuit recovery.

The [uncued rule-edit weight-prior test](ENDOGENOUS_EDIT_WEIGHT_PRIOR.md) examines grammar-valid induction and place edits in the six-frame joint language. At the later attention bond, an eight-direction basis from verb-output weights improves induction mediation over generic weight SVD on held-out sentence frames but remains worse than an edit-fitted basis; it fails to help the previous-place rule. First-bond patch controls explain why a tidy local subspace there would not establish the model's full rule computation.

The [joint-rule basis transfer test](JOINT_RULE_BASIS_TRANSFER.md) fits one local basis on several edit families and, separately, leaves each family out of basis fitting. Across two frame splits, pooling known edits can mediate several rules at matched width, but the leave-one-rule-out basis transfers poorly, especially to the previous-place dependency. This gives a concrete whole-rule-family holdout for future shared-dictionary featurizers; current results remain supervised local mediation, not unlabeled rule discovery.

The [shared-group featurizer test](SHARED_GROUP_FEATURIZER_TOY_LANGUAGE.md) fits the existing overlapping dictionary/gate architecture on lexical and induction contexts, then tests the held-out previous-place edit family on two new stream panels. Scalar Top-8 wins both reconstruction and local causal-patch fidelity; an explicit membership-coverage cost fixes atom collapse but not the gap. A matched paired activation-change objective at weights 1 and 5 improves scalar Top-8 more than groups. This bounded negative result keeps the held-out edit family distinct from model-training holdout or genuine new-rule discovery.

The [dependency-proxy audit](DEPENDENCY_PROXY_REALISM_AND_REPEAT_TRADEOFF.md) compares repeat/copy opportunities and bracket spans against the frozen FineWeb development panel. Its raw successor-copy rate was dominated by fixed phrase continuations; conditioning on a lexical successor exposes the tradeoff between lower repetition and induction task density. A new distant-subject-return setting restores most induction cases at longer range. Its matched bilinear model improves long-distance causal verb edits over the original local-repeat model but worsens short-distance edits on two independent panels; aggregate induction error is nearly unchanged. Improving naturalness still requires more varied frames and lexical content.

The [short-to-long featurizer transfer test](INDUCTION_DISTANCE_FEATURIZER_TRANSFER.md) uses that return-model checkpoint at the final attention bond. A full local patch recovers the model's long-distance edit effect, and an eight-direction short-edit basis constrained to a weight-derived 32-dimensional prior transfers across distance, frames, and test streams better than raw weight SVD, activation PCA, or short-response quartimax. The native model remains far from the exact long-range oracle, so this is local model-effect retention, not a recovered rule circuit.

The [readable surface-mixture experiment](SURFACE_MIXTURE_RESULTS.md) introduces a 141-token grammatical stream with four competing verb rules and an exact prefix oracle. It includes a quantitative and manually reviewed FineWeb calibration, 1–4-layer bilinear training, paired causal edits, and attention-bond gauge/truncation tests. Its realism audit still identifies a large gap to web text.

The [proposal alignment](PROPOSAL_ALIGNMENT_2026-11-02.md) distinguishes the original joint-rule interference objective from task-family transfer. The [multi-genre experiment](MULTI_GENRE_RESULTS.md) adds reports, numbered guides, FAQ copying/recall, and conditional policies; it audits an exact prefix oracle, trains 1–4-layer attention-only bilinear models, and tests causal edits and local bond bases with held-out object words. It is still much less diverse than length-matched FineWeb, and its separate genre headers do not replace joint-rule competition.

The [conditional readable-rule experiment](CONDITIONAL_SURFACE_RESULTS.md) keeps competing lexical, induction, lead and place rules in one stream and changes their weights through a visible mode cue. It tests exact fixed-prefix probability shifts, sampled versus oracle-soft supervision, a nearly two-rule control, and local attention-bond mediation. Its principal negative result is that compact cue transmission and low next-token KL do not yet imply the correct context-dependent induction effect.

The [six-frame surface-mixture follow-up](SURFACE_MIXTURE_V2_RESULTS.md) broadens the same joint-rule data-generating process across sentence forms, audits every prefix against its exact oracle, and adds grammar-valid rule-specific causal edits. Its sentence-length distribution is closer to a fixed FineWeb sample, while lexical diversity and naturalness remain far behind.

The [two-site joint-rule follow-up](SURFACE_MIXTURE_V3_RESULTS.md) adds a second contested prediction at object nouns within the same six-frame stream, with adjective bigram, prior-object induction, frame cue, and previous-event proposals. It audits both exact mixtures, trains bilinear models on both, and tests whether verb-trained response directions transfer to the object task. The generated prose remains much less diverse than FineWeb.

The [joint narrative and question follow-up](SURFACE_MIXTURE_V4_RESULTS.md) keeps those two contested prediction sites and adds object-keyed retrieval questions in the same stream. It audits exact answers, harder questions with intervening sentences, surface forms, and supported causal edits. The latest-place shortcut and residual form gap are explicit baselines.

The [joint subject-count follow-up](SURFACE_MIXTURE_V5_RESULTS.md) adds a second question type to that same narrative stream: count how often a named person is the main actor. It preserves the verb/object mixtures and object-place questions, audits the exact count oracle and alternative-key answers, and evaluates count-specific causal edits separately from place retrieval.

The [FineWeb form/task notes](FINEWEB_FORM_TASK_NOTES.md) record a small, form-stratified spot check of the frozen panel and map observed quotes, questions, headings, numbers, aliases, and corrections to proposed exact-oracle task families. This is hypothesis generation, not a prevalence estimate or a claim of realistic synthetic text.

The [inspected FineWeb dependency panel](FINEWEB_DEPENDENCY_PANEL.md) pins 25 cue-selected windows and manually distinguishes actual local source binding, corrections, aliases, numeric updates, and numbered-item scope from regex false positives. It informs the versioned [v6 source-binding stream](SURFACE_MIXTURE_V6_RESULTS.md), which interleaves quoted claims and source questions with the joint verb/object rules; the task-rich sampling rates are deliberately audited against FineWeb rather than presented as realistic prevalence.

The [random-window FineWeb form check](FINEWEB_RANDOM_FORM_PANEL.md) reviews 32 additional windows from the disjoint development panel without selecting on task cues. It shows far more document and parenthetical variety than the six-frame stream; the richer low-rate v8 stream matches some coarse cues but overproduces colons and still lacks parentheses. The [parenthetical-alias experiment](SURFACE_ALIAS_RESULTS.md) implements an alias defined in an aside and later used as a subject of the same lexical/induction collision. Its exact-prefix oracle and paired edits pass audits. A lower-rate setting matches parenthesis frequency on a separate FineWeb panel, but the overall surface still differs sharply from natural text. The trained bilinear pilot partly responds to earlier alias definitions but fails alias-to-canonical-name equivalence; two matched causal-supervision runs worsen both aggregate prediction and edit fidelity. Stronger two-definition curricula and a lexical-dominant softmax control also fail alias substitution. On held-out key-switch pairs, their predictions are much closer to an equal mixture of both referents' verb distributions than to the correct referent's distribution. Swapping alias labels between two definitions almost never changes the lexical-control model despite nearly maximal oracle effect, while reversing definition order also leaves it invariant. This pooled-output shortcut motivated the later same-vocabulary curriculum transfer, whose competent joint-language checkpoint and remaining limitations are reported below.

The [minimal alias-lookup control](MINIMAL_ALIAS_LOOKUP_RESULTS.md) tests whether the architecture can learn keyed binding outside the joint-rule stream. Across five seeds on exhaustive, disjoint splits, two-layer bilinear models average 92.5% accuracy on held-out alias–person pairings, while the matched-shell softmax controls average 35.6%. The bilinear models follow held-out query and alias-label edits far better. This is a capability and inductive-bias diagnostic on an artificial answer-only task; the subsequent same-vocabulary transfer into the joint language is reported below.

The [minimal-alias bond-basis comparison](MINIMAL_ALIAS_BOND_BASES.md) evaluates local weight-SVD, output-weight, activation PCA, intervention-response, random and coordinate bases on those competent bilinear checkpoints. Eight of 64 local weight directions preserve nearly all held-out lookup behavior at the final bond across five seeds, but an output-weight basis does too: the small answer space explains much of that compression. A first-bond weight basis outperforms the output-weight baseline at the same width. Neither result establishes a human-legible binding circuit. The [same-vocabulary curriculum transfer](ALIAS_CURRICULUM_TRANSFER.md) now yields a competent two-layer model inside the joint-rule language, including held-out alias–person pairings, and tests complete-bond mediation and several local basis priors. It remains a one-seed diagnostic rather than recovery of the original proposal's full competing-rule vocabulary.

The [disjoint-panel check](FINEWEB_DISJOINT_PANEL_CHECK.md) repeats the v6 realism audit on new FineWeb token windows and reviews a second cue-selected dependency sample. It finds that matching quote frequency nearly eliminates source-answer examples in short streams, while numbered-item scope remains a clearer natural dependency candidate. The next version should separate natural-rate generation from conditional training of rare, diagnostic tasks.

The [section-scope experiment](SURFACE_MIXTURE_V7_SCOPE_RESULTS.md) adds exact-oracle scoped facts and section-key edits to the joint stream. V7 exposes a double-number surface confound inherited from generic numbered narrative lines; v8 puts headings on their own lines while retaining other numbered prose. Two v7 bilinear pilots, a clean-format v8 bilinear model, and a same-shell softmax-attention control all remain key-blind under causal edits. These checkpoints are not scope circuits to decompose; interpretation work returns to competent tasks.

The [multi-genre token-tensor experiment](MULTI_GENRE_TOKEN_TENSOR_CANONICALIZATION.md) constructs a weight-only first-layer attention function on two competent FAQ-recall models. A full relative-lag table closely replays native logits; per-lag score SVDs compare data-free and Reports/Policy-weighted compression on fresh FAQ/Guide holdouts and causal edits. The data-aware method improves in-fit Policy fidelity but has seed-dependent FAQ transfer, so tensor compression alone is not a recovered rule component.

The [direct singular-mode ablations](MULTI_GENRE_SINGULAR_MODE_CAUSALITY.md) use separate head-selection, rank-hypothesis, and replication panels. In one model seed, removing a weight-only second singular term across relative lags sharply damages FAQ recall while barely affecting Policy or FAQ copying; removing the highest-energy term does not. This is selective causal importance in that trained function, not yet a human-readable or cross-seed-canonical component.

The [sparse-gauge whole-genre holdout](MULTI_GENRE_SPARSE_GAUGE_HOLDOUT.md) fits task-blind activation PCA/varimax, weight bases, and native-coordinate gates on competent three-layer multi-genre models. FAQ and Guide are withheld from featurizer fitting. Activation varimax improves bond sparsity and post-fit axis purity, but native coordinates better preserve held-out FAQ-recall behavior and causal effects in both trained-model seeds.

The [shared-dictionary/group whole-genre holdout](MULTI_GENRE_SHARED_GROUP_HOLDOUT_RESULTS.md) tests scalar Top-K, fixed disjoint groups and learned overlapping groups at that final bond. On two trained-model seeds, group models lose both in-fit Policy fidelity and held-out FAQ causal effects to scalar Top-16; exact downstream KL in the fitting objective does not consistently repair transfer. Native 16-coordinate gates remain the strongest FAQ-recall intervention. This closes the reconstruction-only group sweep at this bond and redirects to multi-rule path-level or causal supervision.

This folder develops explicit, auditable language tasks for training and analyzing 1–4-block bilinear transformers. The immediate goal is **correct data-generation rules** and diverse causal targets. It is not yet a natural-language corpus or evidence that a model learned a human linguistic abstraction. Training uses the original `DeepModel` implementation in [`tensor_language/deep_model.py`](https://github.com/loganriggs/tensor_language/blob/main/deep_model.py): each block has bilinear attention plus a bilinear MLP. We use RMSNorm for numerical stability, so the trained network is not globally polynomial despite its bilinear modules.

For the current joint-rule language with varied determiners and parenthetical asides, see [the exact-rule, realism, training, and causal-edit results](SURFACE_MIXTURE_V2_VARIED_RESULTS.md). The trained two-layer checkpoint is in `surface_mixture_v2_varied_soft_models/`.

The [matched-width basis transfer test](VARIED_JOINT_RULE_BASIS_TRANSFER.md) compares weight-derived, activation, intervention-trained and sparse-rotated directions on that checkpoint with held-out sentence frames and a parenthetical-rate shift. It separates local response compression from recovery of a rule computation.

The [iterated L1/prune transfer test](VARIED_ITERATED_SPARSITY_PATH_RESULTS.md) applies the earlier revert-and-no-L1-finetune protocol to this varied language model. It reaches 14.23% attention-weight density or 37.72% all-linear density at bounded KL, then tests causal edits, head paths, and an exact first-layer token-space QK/OV fold. Weight sparsity does not yet produce comparably legible rule paths.

The [first-layer functional sparsity test](VARIED_FUNCTIONAL_SPARSITY_RESULTS.md) compares a uniform-vocabulary and a generated-data-weighted penalty on the exact QK1×QK2 token-pair function against an ordinary finetuning control. Uniform weighting markedly concentrates scores but sends most of the largest lag-one scores to pairs unseen in held-out generated streams; distribution weighting reduces that failure without revealing a cleaner causal rule circuit.

The [bidirectional head-path patching assay](VARIED_HEAD_PATH_PATCHING_RESULTS.md) locates much of the original model's lexical effect in the first attention block, induction in the second, and previous-place in second-block head 1 across two independent edit panels. Neither score-sparsity arm sharpens these causal assignments.

The [second-layer source and factor assay](VARIED_SECOND_LAYER_SOURCE_CIRCUITS.md) resolves the mixed head further: second-block head 1 reads the edited previous-place token directly, while head 0 (and partly head 1) reads the earlier verb for induction. Counterfactual QK/V swaps distinguish value content from routing and expose a strong multiplicative interaction for the previous-place rule.

The [matched return-distribution finetune](VARIED_RETURN_FINETUNE_RESULTS.md) tests why long-range induction fails despite source localization. Return-focused training strengthens the long-distance causal effect and its source specificity while weakening short-distance effects; a mixed subject distribution gives an intermediate tradeoff. This produces distance-stratified causal targets for later path-level featurizer comparisons.

The [output-aware shared-group test](VARIED_CAUSAL_GROUP_FEATURIZER_RESULTS.md) keeps previous-place interventions out of featurizer fitting. Shared groups still lose to scalar Top-8 under matched active-support budgets, motivating a direct comparison with the original iterative L1-and-prune path analysis.

## What existed, and what this adds

The older repository already has natural-text training (`lm_train.py`), bilinear model code (`model.py`, `deep_model.py`), and analyzed first-layer/late-layer circuits for bracket closure, numbered-list increment, and successor sequences (`basis_aligned/qk_mdl/algo_tasks/SYNTHESIS.md`). The recent canonicalization toys in this repository tested count, XOR, AND, and shared pair routing. These are valuable circuit probes but cover a narrow slice of language. This suite adds 12 explicit task families, balanced sampling, structural variants, oracle checks, FineWeb form audit, and a 1–4-block trainer.

## Current generated tasks

Each example contains an observable prefix, an exact next-token target, alternatives, and a latent world-state record. `variant=0` is for training; `variant=1` changes surface formulation or structure. Inputs are small word-token sequences, not GPT-2 tokens. The target distribution is chosen by the generator, often deliberately balanced, and **does not reproduce FineWeb frequency**.

| Family | Rule determining next token | Main latent dependency | Structural shift |
|---|---|---|---|
| Agreement | Singular subject → `is`; plural → `are` | Head noun number, ignore intervening noun | Relative clause instead of prepositional phrase |
| Reflexive | Matrix subject maps to `himself`/`herself`/`themselves` | Binding across intervening noun | Relative-clause intervener |
| Attribute binding | Answer the color associated with a queried owner | Two name–color records; event order varies | Possessive paraphrase |
| State update | Latest move of the tracked object determines its location | Object identity and last relevant event | Alternative final-state query |
| Numbered list | Next line number is start + number of items | Arithmetic successor and format | Three items instead of two |
| Weekday successor | Next weekday wraps modulo seven | Ordered lexical cycle | Three days instead of two |
| Delimiter stack | Close the remaining outer opening delimiter | Stack, depth, bracket type | Depth three instead of one/two |
| Comparison chain | A > B, B > C implies top/bottom | Transitive order | Alternative query phrase |
| Conditional rule | Apply the matching color→gate-state rule | Rule selection rather than last-token copying | Compressed rule wording |
| Speaker binding | `I` in a quoted report resolves to that report's speaker | Event speaker/action binding | Question-answer phrasing |
| Correction | Later positive location overrides a wrong or old one | Negation or update, then ignore an unrelated object's newer location | Temporal correction instead of explicit negation |
| Role reversal | Active/passive forms preserve agent and recipient | Semantic roles vs linear position | Passive clause instead of active |

The exact sample space and lexical choices are in `grammar.py`. The supervised target is deterministic **within this controlled micro-world**. Some forms are intentionally simplified or awkward (especially possessive paraphrase, delimiter closure, and explicit role labels). Naturalness is a separate measurement; oracle agreement alone does not certify it. The generator also does not implement ambiguity, pragmatics, or realistic lexical frequency.

## FineWeb calibration

`audit_fineweb_surfaces.py` decodes the existing frozen 480 × 513 GPT-2-token FineWeb panel in the older repository and records surface-cue counts in `fineweb_surface_audit.json`. Parenthesis pairs occur in 309/480 windows, quote pairs in 311, pronouns in 421, numbered-list lines in 33, weekdays in 76, and a state-change-verb regex in 110. These windows are **not documents**. In sampled matched contexts, numbered-list hits often were genuine outline items, but weekday hits were commonly dates or travel advice rather than cyclic continuation; `put` often appeared in idioms or instructions rather than entity-state updates; comparative regex hits often lacked the three-entity transitive chain in our toy. Thus the matches do not tell us whether a target dependency is present, unambiguous, or predictable. We use them only to avoid building a suite entirely divorced from web-text forms. A next audit should manually annotate stratified contexts and separate actual syntactic/semantic dependencies from accidental regex hits.

## Literature-guided additions to build next

These are candidate rule families, **not implemented/validated yet**. Each should have an independently testable oracle, minimal contrast, controlled distractors, and held-out compositions.

1. Determiner–noun number agreement and irregular plural morphology.
2. Longer subject–verb dependencies with two or three attractor nouns.
3. Negative-polarity-item licensing (`any`/`ever`) with scope constraints.
4. Filler–gap licensing and extraction islands; requires expert template review.
5. Pronoun antecedent with gender/number cues and explicit ambiguity rejection.
6. Relative-clause attachment and nested clause role assignment.
7. Quantifier scope in a restricted, explicitly scoped grammar.
8. Negation scope over a conjunction or conditional.
9. Cross-sentence coreference and event continuity.
10. Induction/copy of an earlier proper name after a repeated context.
11. Multi-step entity tracking with swap, containment, and irrelevant events.
12. Temporal ordering (`before`, `after`, `until`) and calendar arithmetic.
13. Spatial relations with perspective changes and multi-hop composition.
14. Set membership, exclusion, and intersection in prose.
15. Discourse contrast/correction (`however`, `instead`) with source reliability.
16. Document structure: headings, bullets, nested numbered lists, and references.
17. Quote depth and speaker shifts in dialogue.
18. Morphological derivation and word-sense disambiguation under context.
19. Arithmetic embedded in receipts, inventories, and comparative prose.
20. Simple semantic parsing with held-out noun–verb and structure combinations.
21. World-knowledge-like rules learned within each document, with an explicit rule table.
22. Long-range retrieval from multiple similar facts, with adversarial distractors.

The grammar-focused families are motivated by [BLiMP](https://aclanthology.org/2020.tacl-1.25/) and [targeted syntactic evaluation](https://aclanthology.org/D18-1151/). Compositional splits follow the motivation of [COGS](https://arxiv.org/abs/2010.05465), [SLOG](https://arxiv.org/abs/2310.15040), and [SCAN](https://arxiv.org/abs/1711.00350). Explicit world-state events follow [Entity Tracking in Language Models](https://arxiv.org/abs/2305.02363). Short naturalistic language with inspectable latent structure is also relevant to [TinyStories](https://arxiv.org/abs/2305.07759), but this suite does not reproduce TinyStories. The FineWeb source is described by the [FineWeb paper](https://arxiv.org/abs/2406.17557).

## Validation and training

```bash
PY=/home/loganriggs/Coding/neuron_circuits/.venv/bin/python
$PY language_task_suite/validate.py
$PY language_task_suite/audit_fineweb_surfaces.py
$PY language_task_suite/train.py --depths 1,2,3,4 --steps 1500 \
  --train-n 12000 --test-n 2400 --d-model 48 \
  --train-variants 0,1 --save-models \
  --output language_task_suite/suite_v1.json
$PY language_task_suite/train.py --objective full_lm --depths 3 \
  --train-variants 0,1 --steps 2000 \
  --output language_task_suite/full_lm_pilot.json
```

`validate.py` checks 240,000 generated examples across all 24 family/variant cells: the target belongs to its candidate set, an independent oracle derived from latent state agrees, and no identical prefix receives conflicting labels. This catches rule bugs and ambiguous labels **within the generator**; it cannot establish human plausibility or lack of shallow shortcuts. Training reports full-vocabulary accuracy, candidate-choice accuracy, each family separately, and the second structural variant. Default training supervises only the final answer token; `--objective full_lm` instead supervises all next tokens in each generated sequence. These are different objectives and must not be compared by raw training loss. `counterfactual.py` tests paired causal and nuisance edits for attribute binding, conditional rules, and state updates using saved checkpoints.

The trainer imports the original bilinear architecture from the local `tensor_language` clone. Set `TENSOR_LANGUAGE_REPO=/path/to/tensor_language` if it is not at the default path in `train.py`. The FineWeb audit accepts `--rows` and `--tokenizer` paths for the same reason; its recorded results identify the exact frozen local panel used here.

**Split limitation:** in the first 12,000-example training / 2,400-example evaluation seed, 90.6% of `variant=0` evaluation prefixes exactly appeared in training. This is because several grammars have tiny finite support (e.g. seven weekday prefixes). That score is a fit check, not evidence of compositional generalization. The `variant=1` evaluation has zero exact-prefix overlap with `variant=0` training, but it bundles wording, structure, and sometimes length changes. Future versions need factorial split controls, new lexical combinations, longer event histories, and a genuinely held-out family test.

For the intended interpretability loop, compare models only once they reach substantial task competence. Then examine whether an attention canonicalization or sparse gauge recovers the known causal variables and rule reuse. Evaluate reconstruction, task accuracy, and counterfactual effects separately. Keep entire **families** out of featurizer training to test new-task discovery; the current variant-1 split holds out formulation, not an entire family. Avoid calling a low-loss sparse representation a recovered computation unless it predicts held-out causal interventions.

A first teacher-distillation comparison of a flat two-layer model, sparse projection factors, and four-layer residual students is documented in [SPARSE_DEPTH_DISTILLATION_PILOT.md](SPARSE_DEPTH_DISTILLATION_PILOT.md). It matches absolute attention-edge budgets and tests both held-out teacher KL and rule-edit fidelity. The deeper identity-initialized student's added blocks remain functionally unused, while an actively split four-layer student loses fidelity under the current pruning schedule. This is a pilot of optimization and representation choices, not yet evidence for or against deeper sparse circuits.

A [weight-only sparse-factor follow-up](SPARSE_FACTOR_WEIGHT_FIT_RESULTS.md) first fits sparse projection products without language data, then distills at a fixed mask. It nearly matches the flat sparse model's responses to known verb-rule interventions at the same edge budget, but its largest intermediate units still affect several rules at once. This motivates causal/path-level objectives and a transfer test on the broader verb-and-object language.

The [two-site verb-and-object transfer test](V3_SPARSE_FACTOR_TRANSFER_RESULTS.md) found that weight-fit sparse factors again approach a flat sparse student on teacher KL and valid edits at the same attention-edge budget. One factor coordinate is causally verb-selective on held-out edits, but flat sparse channels can be similarly selective. No evidence yet shows that factorization yields a uniquely simpler or more human-readable circuit.

The [sparse-depth participation test](SPARSE_DEPTH_PARTICIPATION_RESULTS.md) checks whether inserted bilinear attention blocks actually compute at matched attention-edge budgets. Global magnitude pruning deletes them; equal per-projection quotas preserve their physical edges but their causal ablations still have effectively zero output effect. After no-L1 recovery, the four-layer arm loses to the flat control on teacher KL and valid verb/object edits. This closes the current depth-pruning schedule as an interpretability route and motivates explicit task-balanced causal/path objectives.

The [varied-language sparse source-path audit](VARIED_SPARSE_SOURCE_PATH_RESULTS.md) exactly replays second-layer `QK1 × QK2 × V` paths and tests dynamic top-K source selection on three independent valid-edit panels for each of two competent model checkpoints. At four retained positions per head, ranking by effect on known verb outputs preserves the dense model's signed rule-edit effects better than ranking by residual-vector reconstruction or generic vocabulary sensitivity, despite worse vector reconstruction. This is a task-output prior for path selection, not sparse parameters, a unique circuit, or better agreement with the exact generator. The next architecture test should use these path targets to train sparse or factored students while preserving whole-rule-family causal-target holdouts and checking that any added depth contributes functionally.

The [sparse path-transfer pilot](VARIED_SPARSE_PATH_TRANSFER_PILOT.md) tests that next training step on matched 29,491-edge flat and factored students. A teacher-gauge path loss substantially reduces held-out selected-path error, including on a cue-edit family omitted from paired fitting, but does not improve teacher KL or signed causal edit fidelity. This rules out the simple fixed-mask path-matching objective as a demonstrated benefit under the tested setup; do not infer that the selected paths are sufficient or that additional depth has been tested by this pilot.

The [source-sufficiency follow-up](VARIED_SPARSE_SUFFICIENCY_PILOT.md) instead distills the dense teacher's output through only four teacher-selected second-layer sources per head. The complete-source executor replays native logits, but the gated output loss overfits sampled edits and does not consistently improve fresh gated predictions; native rule-edit fidelity worsens in both a short and a larger-data run. This closes the fixed teacher-top-four loss on the varied-language checkpoint and redirects work to broader task coverage and genuinely learned or independently compositional sparse circuits.

The [task-balanced sparse causal-transfer test](V3_SPARSE_CAUSAL_TRANSFER_RESULTS.md) resumes flat and factored fixed-mask students with a loss on teacher probability changes under valid verb/object edits, withholding the entire previous-event object edit family from paired supervision. At coefficients 0.1 and 0.01, supervised-family validation and omitted-family teacher-effect errors are worse than KL-only recovery; exact-oracle effects and aggregate KL show different tradeoffs. The gradient path passes a fixed-batch check, so the next objective needs better conditioning or explicit path-level constraints. This is an edit-supervision holdout, not an unseen model-training task.

The [versioned Catalog retrieval experiment](MULTI_GENRE_V2_CATALOG_RESULTS.md) adds object-keyed two-field records and several question forms to the multi-genre stream. Its exact oracle and four causal edit families pass an independent audit, but 1–4-layer bilinear models, a softmax attention control, and first-answer-weighted training all fail a query-key-blind Bayes baseline and the key-edit test. The task is therefore a useful capability gate, not yet a recovered circuit. Its FineWeb form audit also shows that the enriched synthetic genre remains much less diverse than web text.

The [joint token-pair factorization test](MULTI_GENRE_JOINT_PAIR_FACTOR_RESULTS.md) extends the first-layer attention tensor comparison with a rank-8 weighted fit to actual query–key co-occurrences. It nearly eliminates in-fit Policy model KL, yet does not consistently preserve held-out FAQ recall or its causal edits across two trained seeds. When FAQ is included during fitting, its model KL becomes very small. This separates distribution-specific compression from transferable or human-readable computation.

The [numeric score-report task](NUMERIC_REPORT_RESULTS.md) follows a second random FineWeb dependency review and adds an exact-oracle person–score binding plus magnitude comparison with four causal edit families. Its surface is still far from web text. Attention-only bilinear models at depths 1–4, an answer-weighted control, and a bilinear-MLP control all stay above a binding-blind Bayes baseline and barely respond to score swaps. It is an audited task family, not yet a competent model to decompose.

## Data-generating process to aim for

A more faithful toy language should sample a **latent document**, not merely a final answer token:

1. Draw a domain/genre (instruction page, list, dialogue, story, explanatory article) and a document length from a measured distribution.
2. Draw entities, attributes, event sequence, and discourse relations from an explicit causal world model. Record every state transition and binding edge.
3. Choose a communicative intent and lexicalize the same world graph through multiple sentence and paragraph structures. Permit incidental details, distractors, and genuinely irrelevant sentences.
4. Produce full text and train ordinary next-token prediction as well as targeted rule probes. The full-LM and answer-only objectives should be reported separately.
5. Construct paired worlds that differ in exactly one causal edge, plus matched pairs that alter only a distractor. Evaluate whether model logits and recovered features follow that edge.
6. Validate generated prose by human judgments and by comparing form statistics with a held-out natural corpus. Corpus frequency should influence the sampling mixture, but not overwrite the exact latent rules. A web-text passage with genuine ambiguity must not be assigned a fabricated unique answer.

This separates **competence on the generated micro-world**, **transfer across surface forms**, and **similarity to real language**. The present suite achieves only the first two in a limited way. FineWeb can calibrate genres, token lengths, punctuation, entity mentions, and distractor prevalence; it cannot supply the latent causal graph automatically. For a strong follow-on, annotate a small stratified FineWeb set for the presence/absence of each dependency, then fit these mixture and distractor rates before scaling the synthetic corpus.
