# A controlled fact-query panel fails native competence

M106, 2026-10-29. All six frozen cohorts failed the competence gate. **This panel cannot currently validate featurizers as representations of correct entity-specific fact retrieval.** No featurizer was fitted or evaluated, and no failed prompt was replaced.

## Why this experiment was needed

M105 found more reproducible whole task effects than individual block roles. A stronger test of reusable computation should independently vary two facts and which fact is requested. Otherwise a model can appear successful by repeating a salient value without selecting the requested entity. This panel was intended to provide controlled causal supervision for that distinction.

Existing task review found that earlier `entity_binding` examples concerned giver/recipient completion, and `object_color` concerned usual-color knowledge. The new panel instead explicitly states two arbitrary facts and queries one. These are different demands. We make no claim that failure here invalidates earlier competence or demonstrates a universal inability to bind entities.

## Frozen factorial design

Three content domains: favorite colors, owned objects and cities of residence. Each has two lexical cases and two formats: sentence completion or question-answer. Independently vary:

- Entity A's specified value between two alternatives.
- Entity B's specified value between two other alternatives.
- Whether the query requests A or B.
- Which fact is mentioned first.

This gives192 prompts, six domain/format cohorts of32 prompts, and384 single-factor edges. All answer continuations are verified single tokens and all four candidate values are distinct. There are no exact prompt-token overlaps with the prior training, source-group or whole-task protocols checked. This is three content domains sharing a retrieval operation, not three discovered tasks or evidence of pretraining exclusion.

The prospective gate requires at least75% four-way accuracy and50% exact full-vocabulary top1 **in each queried-entity × mention-order cell**, with eight prompts per cell. A cohort passes only if all four cells pass. This blocks success based only on one entity or one mention position.

Twenty-four prompts were provisionally designated for possible later training: lexical case0, completion format, diagonal fact combinations a=b. The other168 reserve combinations, lexical cases or format. The competence failure means none is promoted as positive semantic-role supervision. No featurizer outcomes were inspected.

## Native results

| Domain / format | Four-way accuracy | Exact vocabulary top1 | Query switches changing candidate choice |
|---|---:|---:|---:|
| Colors / completion | 3.1% | 3.1% | 1/16 |
| Colors / question-answer | 46.9% | 0.0% | 1/16 |
| Objects / completion | 43.8% | 43.8% | 2/16 |
| Objects / question-answer | 46.9% | 46.9% | 1/16 |
| Cities / completion | 37.5% | 6.3% | 4/16 |
| Cities / question-answer | 43.8% | 0.0% | 0/16 |

All six cohorts fail. Across96 query-switch pairs, the preferred candidate changes only9 times; only one pair has both requested answers correct. Candidate-choice stability does not imply that the full output distribution is unchanged.

Four-way scoring ranks the four declared canonical continuations. Full-vocabulary top1 is stricter and can be sensitive to answer syntax or alternative wording, especially in question-answer prompts. But the low four-way accuracy and rare appropriate query switches also fail to support intended retrieval, independently of that stricter top1 requirement. These are fixed-panel measurements, not population estimates.

## Post-result error description

This accounting retains every cohort and does not fit a heuristic or alter the prompts:

- Color completion picks an **unmentioned** candidate color in29/32 cases. For example, after specifying red and blue, several prompts favor green regardless of which person is queried.
- Color question-answer and both object formats always pick one of the mentioned candidates, but frequently pick the other entity's value. Their query-switch accuracy is poor even though their four-way accuracy exceeds chance.
- City formats mix queried, other-entity and unmentioned candidate choices.

These observations suggest that value availability, generic continuation preferences and entity selection need separate tests. They do not establish a particular native circuit, prove that the model implements a named heuristic, or justify relabeling errors as successful new tasks.

## Research consequence

Do not use intended correct-retrieval labels here to assert that a featurizer should uncover native retrieval units. Training such a label could inject behavior that this model does not reliably exhibit. Native-response fidelity could still be measured on errors in another explicitly scoped study, but it would answer a different question.

This failure is useful for the research loop: broad-looking task labels are insufficient. We need demonstrated dependence on the relevant factor and invariance to irrelevant factors before interpreting a learned group as a reusable task component. The native competence check prevented a misleading downstream feature-validation experiment.

Close this screen without prompt or threshold search. Next review existing competent task/control evidence for a context-dependent operation with a discriminating nuisance control, rather than continuing to repair this panel. Any future compositional task proposal needs its own prospectively frozen competence criterion. Preserve this panel as a failed benchmark, and preserve all existing whole-task training exclusions. New-task discovery remains unachieved.

## Verification and saved data

198 native forwards:192 prompts plus six direct-native-forward checks against the instrumented trace. The pinned model checksum is unchanged. All192 final logits, exact token IDs and normalized inputs/product writes at MLP0/1/2 are saved locally in `emlp_m106_fact_query_states.pt`.

An independent audit reconstructs prompt labels and factorial coverage, checks every answer tokenization and data role, independently rescales all final logits and verifies all24 cell gates/six cohort gates. It checks all384 factor edges: every fact-A, fact-B and query edge is token-length aligned (96 each); only48 of96 mention-order edges are aligned. No internal intervention was performed. Maximum metric discrepancy8.33e-17. Early-state caches were checked for shape and finiteness, not independently reexecuted.

- [Frozen protocol and all prompts](validation/fact_query_panel_protocol.json)
- [Native results and competence gates](validation/fact_query_panel_screen.json)
- [Independent audit](validation/fact_query_panel_audit.json)
- [Explicitly post-result failure accounting](validation/fact_query_panel_failures.json)
