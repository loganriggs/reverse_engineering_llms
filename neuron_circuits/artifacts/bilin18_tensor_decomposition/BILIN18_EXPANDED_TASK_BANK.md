## H962 — Expanded supervised task bank: sixteen candidates, eight discovery-competent families

Priority follows the user's task-bank proposal. The earlier broad screens already found reliable agreement, entity binding, capital recall and some sequence/sentiment formats; arithmetic, temporal/relational instructions and several other formats were unreliable. This run uses a fixed sixteen-family paired screen rather than assuming every described task is available in this FineWeb-trained model. H961 context bridges are complete; their frozen predictor evaluation remains pending, superseded in priority by task-bank expansion.

256 declared pairs,16 families,8 source slots,2 formats. Format A sources0–3 are discovery and4–7 lexical candidates; format B mirrors those cohorts for structural/joint candidates. These are slot-based development splits, not universally vocabulary-disjoint partitions: cyclic colors/countries and shared task terms cross cohorts. Before encoder training, compatible families need stricter grouping to prevent overlapping content from being advertised as lexical independence. Role binding, spatial inversion, sequence copy and tense were preassigned whole-task training holdouts. Their competence screen is visible, but they must not be used to train a proposed multi-task featurizer.

Fixed gate: >=4 distinct valid pairs per cell, >=75% pairs correct at both binary endpoints, >=50% intended semantic-group full-vocabulary top1. All failures and exclusions retained. Exact tokens used except inherited fixed agreement/sentiment/negation groups. No equal-length requirement for native competence; alignment metadata is reported separately.

| Family | Discovery | Lexical candidate | Alternate format | Joint candidate | Whole-task holdout |
|---|---|---|---|---|---|
| agreement | pass (100%/100%) | pass (100%/88%) | pass (100%/100%) | pass (100%/75%) | False |
| arithmetic | fail (25%/0%) | fail (0%/17%) | fail (0%/0%) | fail (0%/0%) | False |
| article_sound | fail (0%/0%) | fail (0%/0%) | fail (0%/0%) | fail (0%/0%) | False |
| attribute_binding | fail (0%/50%) | fail (25%/62%) | fail (0%/50%) | fail (0%/38%) | False |
| capital_recall | pass (100%/75%) | pass (100%/100%) | pass (100%/100%) | pass (100%/100%) | False |
| entity_binding | pass (100%/100%) | pass (100%/100%) | pass (100%/100%) | pass (100%/100%) | False |
| key_value | fail (0%/0%) | fail (0%/0%) | fail (0%/0%) | fail (0%/0%) | False |
| negation | pass (100%/50%) | pass (100%/50%) | fail (100%/25%) | fail (100%/25%) | False |
| numerical_comparison | fail (0%/0%) | fail (25%/0%) | fail (0%/0%) | fail (0%/0%) | False |
| pronoun_gender | pass (100%/100%) | pass (100%/100%) | fail (100%/25%) | fail (100%/25%) | False |
| role_binding | fail (0%/25%) | fail (0%/12%) | fail (50%/75%) | fail (50%/62%) | True |
| sentiment | pass (100%/75%) | pass (100%/75%) | pass (100%/75%) | pass (100%/75%) | False |
| sequence_copy | pass (100%/88%) | pass (100%/75%) | fail (100%/12%) | fail (100%/38%) | True |
| spatial_inverse | pass (100%/100%) | pass (100%/100%) | fail (0%/0%) | fail (0%/12%) | True |
| temporal_order | fail (0%/25%) | fail (25%/12%) | fail (25%/62%) | fail (25%/62%) | False |
| tense | fail (100%/25%) | fail (100%/38%) | fail (100%/12%) | fail (100%/38%) | True |

Percentages are both-endpoint binary correctness / intended full-vocabulary top1. Four families pass all tested cells: agreement, capital recall, entity binding, sentiment. Four additional families pass discovery and lexical candidates: negation, pronoun gender, sequence copy, spatial inversion. Their alternate formats fail at least one gate. These failures are not proof the model cannot perform the abstract task; prompt formulation, answer conventions, and task changes matter. In particular spatial A uses lateral inversion while B uses vertical inversion, so this is not a pure wording transfer test.

The strongest additional eligible mechanisms beyond agreement/sentiment are entity binding and factual recall, plus pronoun gender in its first format. Sequence copy and lateral spatial inversion provide predeclared held-out-task candidates. Negation is competent in A but changes input length; do not force it through equal-absolute-position patching without defining a valid alignment. The proposed ten-plus candidate search is practical, but this model does not yield ten-plus usable mechanisms merely by naming them.

Two arithmetic pairs are excluded by strict answer-token boundary checks;254 valid pairs/508 endpoints. Accepted run509 model forwards including a repeat check; deterministic logits replay within2.34e-5. An input-only revision happened before inspecting first-run outcomes: spatial/pronoun rows repeated prompts despite a four-distinct-pair gate, and article prompts had trailing spaces inconsistent with the continuation convention. Source/manifest/outcomes/log of that preliminary run are preserved; final manifest has256 distinct prompt pairs. All compute from both runs is accounted in JSON audits. No output-based prompt replacement or threshold change.

Independent manifest/gate recomputation passes; input and manifest hashes unchanged; finite outcomes; no training, causal localization, or new natural documents. Reserved test untouched. This screen establishes behavioral eligibility only, not meaningful manifolds or causal units.

Next: inexpensive causal localization on discovery examples in eligible families, including held-out-task diagnostics kept out of featurizer training. Capture last-token MLP writes, scan fixed candidate layers, retain exact-output KL and signed task effects, and use identity/norm-matched controls. Choose sites only on discovery and freeze them before evaluating other cohorts. Add task-appropriate alignment and semantically same-answer controls before expensive Fisher targets. Explicitly maintain task-level training holdouts; do not optimize a universal layer4 featurizer just because that site worked for agreement/sentiment.
