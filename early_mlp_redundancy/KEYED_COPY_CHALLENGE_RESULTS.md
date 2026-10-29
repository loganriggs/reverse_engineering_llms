# Successful repetition completion does not establish keyed retrieval

M107, 2026-10-29. The historical one-key sequence-copy examples replay exactly. Both fixed two-key challenge formats fail native competence. We therefore retain the original result as evidence for **one-key repetition completion**, without promoting it to general key-conditioned retrieval.

## Why this is a distinct test

Reviewing the existing competence evidence revealed that a prompt such as `white red; white red; white` can be solved by copying a mentioned value without selecting among competing keys. M106 tested arbitrary facts in natural-language completion/question-answer formats and failed; it remains closed. M107 instead tests a loophole in an already positive sequence-copy benchmark using its existing word/punctuation family.

The alternatives were fixed before new outcomes: use two keys with independently varied values, repeat both pairs twice, and switch the requested final key and pair order. For example:

`white red; orange blue; white red; orange blue; white`

The counterpart ending in `orange` requests blue instead. Pair order and both values vary independently. This is a selective-retrieval challenge, not a claim that every such language-model completion has a unique valid continuation.

## Frozen design and gate

Sixteen original endpoints are exact replay controls. Sixty-four challenge prompts use two lexical configurations, semicolon/comma formats, two values per key, two final queries and two mention orders. All answer groups consist of verified single-token original/capitalized/uppercase variants and are disjoint.

For each format, every query × mention-order cell must achieve at least75% four-way candidate accuracy and50% vocabulary top1, with eight prompts per cell. A format passes only if all four cells pass. This prevents a pure first-value or last-value predictor from passing by doing well on just one query/order condition.

The entire panel remains excluded from featurizer fitting under the existing sequence-copy whole-task exclusion. There is no new parameter fitting, prompt replacement, threshold adjustment or candidate selection. The two-key contexts are longer than historical controls, so this is not a length-matched causal comparison.

## Results

| Panel | Candidate accuracy | Intended vocabulary top1 | Gate |
|---|---:|---:|---|
| Historical one-key controls | 100% (binary) | 50% | Exact replay |
| Two-key semicolon | 59.4% (four-way) | 46.9% | Fail |
| Two-key comma | 56.3% (four-way) | 25.0% | Fail |

The candidate sets differ between historical and challenge rows; the percentages are not matched-difficulty estimates. The challenge results fail the independently specified gate, while historical baseline identity verifies that the model/runtime has not changed.

Post-result descriptive accounting among the four declared candidate values:

| Format | Requested key's value | Other key's value | Unmentioned candidate | First-mentioned value | Last-mentioned value |
|---|---:|---:|---:|---:|---:|
| Semicolon,32 prompts | 19 | 6 | 7 | 19 | 6 |
| Comma,32 prompts | 18 | 8 | 6 | 24 | 2 |

The first three columns partition predictions; first/last are overlapping alternative descriptions, not additional categories. These candidate choices are not necessarily full-vocabulary top1 predictions.

Switching the final key changes candidate choice in9/16 semicolon pairs and8/16 comma pairs. Both requested answers are correct in4/16 and2/16, respectively. Thus there is partial query sensitivity; the result does not support complete key blindness. Preference for first-mentioned values also argues against describing all behavior as simply copying the most recent value. These are behavioral descriptions, not localized circuit explanations.

## Consequences for computational units and supervision

A correct label on a simple task can support several computations. The original one-key benchmark did not distinguish position-based copying, repetition/frequency effects and key-conditioned retrieval. The new challenge limits the stronger interpretation but does not uniquely identify a shortcut: sequence length, interference and partial key sensitivity all remain possible contributors.

Feature supervision should describe the behavior actually demonstrated by the native model. A featurizer that predicts correct two-key retrieval on these examples would not thereby explain the model's existing computation. Conversely, reproducing native errors is a legitimate fidelity target if it is explicitly labeled that way, rather than called correct semantic retrieval.

Close M107 without another prompt variant. The next proposed bounded experiment is a fixed factorial sequence panel with one repeated key and three possibly conflicting values. Compare observed native behavior with first-value, last-value and majority hypotheses, and characterize output interactions across the full binary cube. Mixed-value prompts need not have a declared correct answer: the target is the native computation itself. Preserve sequence-copy training exclusions; any later featurizer comparison must use frozen models. First check prior repetition/induction evidence to avoid duplicating a completed study. No such new panel or intervention has been run yet.

## Verification

80 native forwards, with all16 historical final-logit vectors replaying bit-exactly. All80 prompts, labels, tokenizations, answer groups and saved final logits are independently audited. All eight competence cells and both cohort decisions match. Maximum margin discrepancy1.78e-15. Candidate-choice/query-switch counts are retained as explicitly post-result descriptions. No native-component intervention or featurizer evaluation occurred.

Full logits and MLP0/1/2 normalized inputs/product writes are saved locally in `emlp_m107_keyed_copy_states.pt`. The capture is available for later explicitly scoped analysis but is not evidence that any early layer performs retrieval.

[Protocol](validation/keyed_copy_protocol.json), [screen](validation/keyed_copy_screen.json), [independent audit and descriptive counts](validation/keyed_copy_audit.json).
