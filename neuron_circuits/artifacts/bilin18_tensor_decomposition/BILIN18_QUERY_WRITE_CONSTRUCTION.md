## H573 — Query construction is distributed; descriptive shares differ from causal effects

Trace all old-template binding and competent copying delimiter cases from H572. Capture final-position attention and MLP writes at layers 0–13, and the residual entering L14 attention normalization. Compare each edited prompt with its base prompt at its own final position. Final token and initial embedding are identical. Each prior write difference is multiplied by the product of subsequent residual lambdas through L14. Their sum reconstructs the edited-minus-base pre-attention residual.

For r0,r1 and RMS denominators s0,s1, normalized input difference is (r1−r0)/s0 + r1*(1/s1−1/s0). Map this exact finite decomposition through L14's first query projection. The normalization allocation depends on choosing the base denominator; it is descriptive, not a unique causal attribution. Query-head normalization and rotary effects remain active in the subsequent gate tests.

Then clamp each of 28 prior final-position writes separately to its base-prompt value, with all other computation live, plus self control. Only the destination position is changed; the earlier selected source key remains unaffected by causality. There are 128 edited prompts and 3,712 interventions. Measure how much contextual gate correction remains: mean absolute distance from the phase-only coefficient, divided by its native value. Smaller ratios mean movement toward the phase-only prediction, not necessarily better task behavior.

| Condition | Lowest residual-ratio single clamp | Residual ratio | Descriptive raw-query projection share |
|---|---|---:|---:|
|binding_old_paren|mlp_10|0.777|0.114|
|binding_old_comma|attn_9|0.790|0.128|
|copying_old_paren|attn_6|0.711|0.015|
|copying_old_comma|attn_7|0.690|0.042|

No single write removes most of the gate correction. The best conditions leave about 69–82%. Descriptive shares and causal effects need not agree: copying-parentheses attention6 accounts for about 1.5% of the raw-query difference projected onto its total change, yet its clamp reduces gate discrepancy by about 29%. Later responses, normalization and gate sensitivity intervene between those measurements. Projection shares can cancel and are not fractions of independent causal mediation.

Binding's attention9 and MLP10 consistently rank near the top. For a locked follow-up, select the two writes with lowest mean residual ratio across punctuation forms separately per task. This gives attention9/MLP10 for binding and attention7/attention4 for copying. Selection is discovery on the old templates, not a validated minimal circuit.

Audits: residual reconstruction max relative error 2.11e−6; raw-query reconstruction 2.13e−6; self margin replay 9.06e−6 and coefficient replay 9.26e−7; finite outputs and source unchanged. Saved write captures. No new prompts or corpus documents. Next H574 tests these fixed pairs and their union on the other templates, retaining both punctuation forms and cross-task pair controls.
