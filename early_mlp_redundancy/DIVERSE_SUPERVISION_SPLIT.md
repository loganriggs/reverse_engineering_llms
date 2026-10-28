# M10: source-input separation exposes limits of task diversity

2026-10-28. Metadata-only audit and prospective training protocol; no new model forwards or fitting. Previous M9 outcomes motivated broader supervision, but no outcomes are used to choose this split.

## Cross-family overlap matters

Grouping only repeated pairs would permit related lexical examples to cross the split. Grouping by any shared changed source-token ID gives25connected components. A stricter rule joins two pairs if ANY changed source token of either pair appears ANYWHERE in the other pair's input prompts, including unchanged context/distractors. This gives17components. Keep both endpoints, all context variants, and each connected component in one split.

Examples: countries connect capital recall and country-language tasks; nouns connect agreement and plural tasks; distractor nouns connect those to pronoun prompts; sentiment words overlap antonym inputs. These dependencies make independent family-wise random splitting inadequate for source-input transfer claims.

A fixed metadata-only RNG(seed2026102810) assigns components, forcing single-component families to training and requiring every multi-component family in both splits. The first feasible proposal is the third proposal. This search uses incidence metadata only, not model performance. It produces66trainingpairs and14evaluationpairs.

| Family | Strict components | Training pairs | Evaluation pairs |
|---|---:|---:|---:|
| agreement | 1 | 12 | 0 |
| sentiment | 1 | 16 | 0 |
| entity_binding | 4 | 6 | 6 |
| pronoun_gender | 1 | 16 | 0 |
| capital_recall | 3 | 2 | 2 |
| antonym | 4 | 3 | 1 |
| plural_noun | 3 | 3 | 1 |
| past_tense | 4 | 2 | 2 |
| country_language | 3 | 2 | 2 |
| number_successor | 1 | 4 | 0 |

Agreement, sentiment, pronoun gender and number successor cannot support a lexical evaluation split under this rule. They are training-only in the next experiment; report their fitting performance separately and do not fill in an invented transfer estimate. Antonym/plural each have only one evaluation pair, so outcomes will be descriptive and coarse.

No changed source token from either split occurs in any task prompt on the other side. Ordinary context tokens may overlap. This is exact tokenizer-ID input separation, not semantic/synonym separation, unseen labels, novel vocabulary for the language model, or unseen words during prior natural-text featurizer fitting. The teacher outputs distributions over the entire vocabulary. All task examples were already examined in earlier research; this remains development data. Original excluded task families are untouched.

## Frozen next experiment, not yet run

Start from the four M8causal checkpoints (scalar/block, seeds71/72), with two arms each. Both receive120freshAdam.0003updates and unit decoder normalization. Natural-only uses an existing natural training document0–31 at every step. Mixed uses the SAME natural document on even0-based steps and a taskpair on oddsteps. Task family schedule has exactly6updates per each of10families; within a family draw only frozen training pairs. The task loss averages each endpoint's native-output KL equally, each endpoint itself averaged over its positions and all50304logits. No answer-label classification loss.

Thus mixed training replaces60natural updates with60taskpair updates. Updates/initialweights are matched, but tokens/FLOPs are not; this is a training-mixture intervention, not a pure additional-data effect. No loss-mixture coefficient search. Both arms use fixed final step, no checkpoint selection. Preserve original mean/scale andactive32.

Evaluate the14source-separated pairs across6families, alltrainingpairs separately, and unchanged natural-text developmentdocs32–63 for forgetting. Compare frozen M8baseline and both arms. Report everyseed/family; no tuning after evaluation. Passing would support broader causal training coverage, not manifold semantics or genuine new-task discovery.

## Audit and reproduction

An independent graph traversal rebuilds source sets directly from differing endpoint token positions, checks the source metadata, recovers the same17components, and verifies zero cross-split source-to-prompt intersections. It verifies pair coverage, family counts, exactly6taskupdates/family/seed, natural document bounds, and allfour initial checkpoint hashes. Model forwards:0.

Run `prepare_diverse_supervision.py`, then `audit_diverse_supervision_split.py`. The saved protocol contains exact schedules and model hashes. Do not regenerate a different split after seeing outcomes and call it the same experiment.

[Split and component receipts](validation/diverse_supervision_split.json), [frozen training protocol](validation/diverse_supervision_protocol.json), [independent audit](validation/diverse_supervision_split_audit.json), [M9 motivation](FEATURIZER_TASK_TRANSFER.md).
