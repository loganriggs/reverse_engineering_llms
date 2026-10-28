## H565 — Physical gap edits validate directional position predictions, not universal magnitudes

Use eight constructed name pairs in binding and repetition copying, two endpoints, and three gap conditions: no insertion, “(as usual)” (four tokens), or “(as everyone expected)” (five tokens). Insert strictly after the selected source name. Prefix tokens and source position remain identical; final token is “to” throughout. The selected causal source key therefore cannot see the insertion. The final destination query can.

Fix L8H3 and L14H4. Evaluate only the 32 base prompts first, then save 192 rotary-only coefficient predictions before evaluating edited prompts. These predictions reuse base query/key content and apply the edited destination's rotary position. No parameters are fit. Native key states are checked against base. Run four local source-bus-edge conditions: self, predicted coefficient, base coefficient, and deletion. All other edges and downstream computation remain live. There are 96 native prompts and 768 interventions.

Every native cell is 16/16 candidate- and full-vocabulary-correct, including both edits. Hence these tests concern mechanism and margins, not repair of an accuracy failure.

| Task | Insertion | Head | Phase prediction error / unchanged-gate error | Gate-change direction accuracy |
|---|---|---|---:|---:|
|binding|usual|L8H3|0.598|100.0%|
|binding|usual|L14H4|1.807|100.0%|
|binding|expected|L8H3|0.053|100.0%|
|binding|expected|L14H4|1.088|100.0%|
|copying|usual|L8H3|0.879|100.0%|
|copying|usual|L14H4|0.357|100.0%|
|copying|expected|L8H3|0.480|100.0%|
|copying|expected|L14H4|0.352|93.8%|

The predicted coefficient-change direction is correct on 127/128 edited source/head comparisons. Magnitude is more variable: for binding L14H4, phase-only prediction is worse than predicting no coefficient change. For binding L8H3 with the five-token edit it is much better. Source-edge substitution produces corresponding output-margin changes, while all candidate accuracies remain perfect. Thus directional gate prediction is supported, but accurate total behavior prediction or a universal distance law is not.

The physical edit changes query context as well as position. Source keys match within 3.06e−5 (batch/float numerical variation); their causal prefixes are identical. The residual prediction error therefore points to destination-query content and its interaction with rotation, subject to that numerical tolerance. This does not identify what semantic property the query represents.

Before inference, a faulty prefix tokenization assertion was corrected and the two-token name Irene was replaced with single-token Helen. No model outcomes informed this correction. The final design and predictions were fixed before edited-prompt evaluation.

Audits: self-margin replay <=1.63e−5; source/manifest/predictions unchanged; all finite. 96 constructed prompts, no new FineWeb documents, total 26,112. Next H566 uses three distinct insertions at each exact token length to expose content variation while holding the phase prediction fixed. Broad goal active and incomplete.
