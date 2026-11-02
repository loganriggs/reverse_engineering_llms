# Grounded toy-language task suite (first pass)

**Current follow-up:** [`WORLD_RESULTS.md`](WORLD_RESULTS.md) documents the newer latent document-world generator, its 32,000-example oracle audit, 1–4-block and larger-model results, causal-edit tests, final attention-bond gauge experiments, and the measured gap to FineWeb surface statistics. The earlier 12-family suite described below remains a simpler baseline. The document-world v1 has a known indefinite-article error (`a orange`); retain its source with its checkpoints and correct it in a versioned successor.

[`V2_RESULTS.md`](V2_RESULTS.md) follows that versioned successor. It lengthens and repairs the document realizations while preserving the latent-world task, checks oracle/edit/split invariants, and retrains the 1–4-block depth sweep. Select the generator explicitly with `--world-version v1` or `--world-version v2` in `train_world.py`, `validate_world.py`, `eval_world_counterfactual.py`, and `world_bond_gauges.py`.

**Research-direction correction:** [`EXISTING_GENERATOR_BRIDGE.md`](EXISTING_GENERATOR_BRIDGE.md) locates the earlier autoregressive mixed-rule generator and trained tensor-transformers in `toy_models_of_tensor_networks/attn_circuits`. Its competing n-gram, induction, and bracket distributions are the primary sandbox for the intended mechanism-discovery loop. The document-world benchmark is a complementary state-tracking stress test.

This folder develops explicit, auditable language tasks for training and analyzing 1–4-block bilinear transformers. The immediate goal is **correct data-generation rules** and diverse causal targets. It is not yet a natural-language corpus or evidence that a model learned a human linguistic abstraction. Training uses the original `DeepModel` implementation in [`tensor_language/deep_model.py`](https://github.com/loganriggs/tensor_language/blob/main/deep_model.py): each block has bilinear attention plus a bilinear MLP. We use RMSNorm for numerical stability, so the trained network is not globally polynomial despite its bilinear modules.

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

## Data-generating process to aim for

A more faithful toy language should sample a **latent document**, not merely a final answer token:

1. Draw a domain/genre (instruction page, list, dialogue, story, explanatory article) and a document length from a measured distribution.
2. Draw entities, attributes, event sequence, and discourse relations from an explicit causal world model. Record every state transition and binding edge.
3. Choose a communicative intent and lexicalize the same world graph through multiple sentence and paragraph structures. Permit incidental details, distractors, and genuinely irrelevant sentences.
4. Produce full text and train ordinary next-token prediction as well as targeted rule probes. The full-LM and answer-only objectives should be reported separately.
5. Construct paired worlds that differ in exactly one causal edge, plus matched pairs that alter only a distractor. Evaluate whether model logits and recovered features follow that edge.
6. Validate generated prose by human judgments and by comparing form statistics with a held-out natural corpus. Corpus frequency should influence the sampling mixture, but not overwrite the exact latent rules. A web-text passage with genuine ambiguity must not be assigned a fabricated unique answer.

This separates **competence on the generated micro-world**, **transfer across surface forms**, and **similarity to real language**. The present suite achieves only the first two in a limited way. FineWeb can calibrate genres, token lengths, punctuation, entity mentions, and distractor prevalence; it cannot supply the latent causal graph automatically. For a strong follow-on, annotate a small stratified FineWeb set for the presence/absence of each dependency, then fit these mixture and distractor rates before scaling the synthetic corpus.
