# A directional token-form response in the reduced nine-head circuit

The [all-head label-transfer test](ALL_HEAD_LABEL_TRANSFER_RESULTS.md) found
coarse query/key token-form enrichment for several exact gauges, but that
was correlational. This experiment asks whether a proposed component changes
the **full 18-layer model's response** to the corresponding token edit.
The effect is measured inside the rank-256 explanatory circuit, whose
responses to these edits are separately checked against the native 546M
model.

## Selection, edits, and controls

Only FineWeb validation documents 15,000–17,500 choose components: among
the ten highest-energy components of each gauge, take the most enriched
for `space_word -> punctuation`. This selects response-varimax component 16
and direct-L1 component 3. The category means a space-prefixed *query* token
and an earlier punctuation *key* token. Neither gauge was fitted using this
label. The rule for choosing the category itself follows exploratory
inspection of the earlier label panel, so this is not a fully preregistered
semantic discovery test.

For each new 128-token document window, find an earlier space-prefixed word
key and a later space-prefixed word query. Keep the query fixed and replace
the key with punctuation or with a different space-prefixed word. The first
panel uses period/comma versus ` the`/` and`; a second, disjoint-document
panel uses colon/semicolon versus ` of`/` a`. Each panel has 16 windows per
corpus across FineWeb, WikiText-103, and TinyStories: 48 windows per panel,
96 total. Documents used in earlier full-model panels are excluded. The
second panel excludes the first causal panel's documents.

All methods start from the **same rank-256 projected model** and remove one
coordinate. A complete deletion of a sparse component is much larger than
one random-axis deletion, so it is diagnostic only. For the main comparison,
scale each candidate's deletion so that its squared nine-head pair-output
change on FineWeb validation pairs equals that of a selected random-axis
control. A second control is a **different-label component in the same
gauge**, also scaled to the same validation pair energy. The same-gauge
response-varimax control is component 165, labelled
`punctuation -> space_word`; its label reverses the query/key roles.

At the fixed later query position, define the model's logit response to an
input edit as \(\Delta L=L(x_{\rm edit})-L(x_{\rm base})\). For a component
deletion, measure \(\|\Delta L_{\rm baseline}-\Delta L_{\rm deleted}\|^2\)
relative to \(\|\Delta L_{\rm baseline}\|^2\), pooling numerators and
denominators over windows. **Specificity** is this normalized quantity for
the punctuation edit minus the analogous quantity for the word edit.
This is a finite difference-in-differences of final logits, not a formal
path-specific mediation fraction; nonlinear downstream interactions remain.

## Full-model result

Specificity advantage over each control on both panels combined, shown in
units of \(10^{-4}\). The interval is a 10,000-draw bootstrap of windows
within each panel and corpus. The final column is the candidate's actual
test-window layer-0 head-output deletion energy divided by the control's;
values below one make the candidate's comparison conservative on edit size.

| Candidate | Control | Specificity advantage ×10⁻⁴ | 95% bootstrap interval ×10⁻⁴ | Test head-dose ratio |
|---|---|---:|---:|---:|
| Response varimax 16 | Random axis | **12.67** | [7.24, 19.99] | 0.54 |
| Response varimax 16 | Same-gauge different label | **10.88** | [5.91, 17.26] | 0.44 |
| Direct L1 3 | Random axis | 6.31 | [3.15, 10.95] | 0.53 |
| Direct L1 3 | Same-gauge different label | 3.01 | [-0.27, 7.41] | 3.11 |

Response-varimax component 16 beats **both** controls on this test. Its
punctuation-edit response-change energy is 0.123% of the model's total
punctuation-edit logit-response energy; its word-edit value is 0.016%.
The effect is small but directionally selective. The direct-L1 component
beats its random control, but its same-gauge comparison is inconclusive:
the interval crosses zero and the candidate causes about three times the
control's test head-output change. Sparse activity alone therefore does not
pass the stronger causal gate.

The two panels agree in direction for response-varimax component 16 on each
corpus. The colon/semicolon panel has a smaller effect than the period/comma
panel. This is evidence for a **narrow punctuation-sensitive response**, not
for a uniform punctuation concept. The saved records contain every window,
edit token, head-output dose, final-logit response, and KL value.

## Does the reduction preserve the native model's edit response?

`flagship_shared_key_input_edit_fidelity.py` replays all 96 input triples
through the unaltered 546M model and the rank-256 baseline. Across both
panels, the rank-256 model's full-vocabulary logit response at the chosen
query has relative
squared error **0.000312** for punctuation edits and **0.000257** for word
edits. Mean native-to-reduced KL at the unedited query is **0.000143**.
The saved fidelity records report base and both edited-input KL separately.
This is a strong fidelity gate for these tested inputs, but it does not prove
that a physical network component corresponds to explanatory axis 16.

The model's external function is invariant under many equivalent internal
descriptions. Deleting a coordinate of our fitted reduced dictionary is an
intervention on that **chosen explanation**. It reveals a reproducible,
predictive component of the function under two token sets and three corpora;
it does not recover a uniquely implemented attention node. The category is
coarse, selected after exploratory work, and lacks a named language task or
behavioral success criterion. The next test should seek a directional
query/key double dissociation or a task-level prediction, with new templates,
calibrated within-gauge controls, and held-out outcomes.

Reproduce with `flagship_shared_key_counterfactual_causality.py` on the two
recorded seed/token settings, `flagship_shared_key_input_edit_fidelity.py`,
`flagship_shared_key_same_gauge_controls.py`, and
`flagship_shared_key_counterfactual_summary.py`. Their JSON artifacts are
`all_head_shared_key_counterfactual_causality_n8_s91-97.json`,
`all_head_shared_key_counterfactual_causality_n8_s101-107_punct25-26_word286-257.json`,
`all_head_shared_key_input_edit_fidelity_n96.json`,
`all_head_shared_key_same_gauge_controls_n96.json`, and
`all_head_shared_key_counterfactual_summary.json` under `full_width_results/`.
