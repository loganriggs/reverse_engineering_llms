## H917 — Crossing label assignment with field order does not establish competent abstract task switching

H916 confounded label meaning with demonstration field order. H917 independently crosses A/B meaning with A-first/B-first records, while keeping the instruction's semantic order fixed: language, then country. Two shared demonstrations use Argentina and Australia; neither occurs in evaluation. All four conditions and all eligible countries are retained. Mapping 0 assigns A to language; mapping 1 assigns B to language.

|Cohort|Record order / mapping|Language exact top1|Copy exact top1|
|---|---|---:|---:|
|Old|A first / 0|68.75%|100%|
|Old|A first / 1|6.25%|100%|
|Old|B first / 0|31.25%|93.75%|
|Old|B first / 1|6.25%|100%|
|New|A first / 0|8.33%|100%|
|New|A first / 1|0%|100%|
|New|B first / 0|8.33%|100%|
|New|B first / 1|0%|100%|

Only old-country A-first/mapping-0 passes the existing joint competence gate. There is no competent counterbalanced mapping pair, and no new-country language cell passes. Field order matters in mapping 0, but reversing the assignment fails in either order. These results do not isolate a pure label bias or establish absence of an abstract task representation; they establish that this particular arbitrary-label protocol is unsuitable for interpreting reliable task switching.

The binary language-answer comparison is misleading here: all old-country pairs and five of six new-country pairs prefer the correct language over the other country's language, despite low full-vocabulary accuracy. Comparing the intended language against the current country's name yields preference fractions equal to the language top1 fractions in this table. The model often prefers the copying answer. This diagnostic is saved separately, and is not treated as a complete readout of internal task selection.

### Decision and next experiment

Stop extending this arbitrary-label prompt rescue sequence. The matched natural-language query evidence from H915 remains evidence for query-conditioned processing, not a proven mapping-invariant semantic operator. Return to competent binding tasks and their matched copying controls. Their weak single-MLP effects in H906 motivate attention/value-bus interventions before fitting another small MLP feature basis.

For the next experiment, separate routing from carried value: use native donor/recipient substitutions at the previously specified reader heads, with identity and full replacement controls. Fit any candidate shared value subspace on discovery name pairs only; evaluate its effect fidelity on the other name pairs and both templates. Retain copying controls and measure full-distribution KL as well as answer margins. This reconnects the featurizer objective to causal structure: a useful manifold must preserve the component's effects and transfer, not merely reconstruct its activations. Head-specific effects should be measured before interpreting low-rank recovery.

### Verification

224 native forwards; 112 within-mapping one-query-token controls and 112 cross-mapping same-query checks pass. Source hashes unchanged; results finite. No intervention or fitted operator in this experiment. No reserved-test evaluation or new natural documents; the document pool remains 26,112. The broad research goal remains active.
