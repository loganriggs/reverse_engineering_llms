## H579 — Suffix carriers differ by head and context

On the reused H572 copying panel, replace contextual values at each of four aligned suffix positions, all four, or all except one; include self controls. Native attention routing stays fixed, and the persistent token-identity bus matches exactly. Positions are yesterday, period, He/She, and spoke/wrote. These 2,560 interventions cover both producer heads, both templates, all four phrases, both delimiters, and every name/variant. No success filtering or new corpus documents.

The statistic below is remaining distance of the downstream L14H4 attention coefficient from the phase-only prediction, divided by native distance. One means unchanged; below one means closer to phase-only. It is neither task accuracy nor a percentage of independently explained computation. These interventions modify the producer's destination write; other downstream paths remain live.

| Template / delimiter / head | All values | Yesterday | Period | Pronoun | Verb |
|---|---:|---:|---:|---:|---:|
|copying_old_paren_L4H0|0.919|0.992|0.954|0.976|0.989|
|copying_old_paren_L7H8|0.858|0.998|0.990|0.946|0.925|
|copying_old_comma_L4H0|0.885|0.974|0.977|0.927|0.954|
|copying_old_comma_L7H8|0.851|0.996|0.970|0.948|0.928|
|copying_new_paren_L4H0|0.764|0.998|0.861|0.931|0.966|
|copying_new_paren_L7H8|0.957|0.998|1.001|0.993|0.965|
|copying_new_comma_L4H0|1.212|1.001|1.091|1.062|1.037|
|copying_new_comma_L7H8|0.932|1.000|1.000|0.985|0.951|

For L7H8, the verb is the strongest individual position in every cell. Selecting the largest absolute departure from one averaged over old-template delimiters picks that position, and it transfers to the new template. The new-template verb replacements yield 0.965/0.951 versus all-values 0.957/0.932; restoring everything except the verb yields 0.991/0.987. This supports a concentrated carrier within this panel, not a universal verb feature.

L4H0 is more distributed. The same old-template selection rule picks the pronoun, narrowly ahead of the period, but the period has the largest effect in the new template. The new-template reversal spans period, pronoun and verb: each restoration moves the coefficient closer to phase-only with parentheses and farther away with commas. Yesterday contributes little there. A single context-independent carrier does not explain this head.

All-except-one interventions are retained in JSON to expose dependence on the other restored positions. Effects need not add after live downstream computation. No confidence intervals treating related prompt variants as independent observations are claimed. Both templates have already been explored in preceding experiments; this is within-panel transfer, not an untouched confirmatory dataset.

Checks: source-group reconstruction <=5.79e-7 relative error; exact suffix identity and bus equality; full-value restoration replays H578 within 1.05e-5 margin; self replay <=2.01e-5; finite coefficients/margins and unchanged source hash. Float32 computation uses bf16-rounded model weights.

Next: trace upstream attention and MLP writers into the verb at L7H8 and period/pronoun at L4H0. Compare position-specific local-value mediation with whole-write effects. Before generalizing, lock a prediction and test a fresh panel spanning copying, person binding, grammatical agreement, factual recall and sentiment, with multiple templates per task, matched nuisance edits, counterfactual answer changes, and whole-family competence gates. Include natural-text coverage separately. Compare contextual-state transport, sentence-boundary effects, grammatical-role encoding, and generic answer preparation; each predicts different transfer patterns. Track internal response, output-distribution KL, and CE separately rather than treating them as interchangeable fidelity targets.

This is progress toward identifying computational units, not completion. Total tracked FineWeb documents remains 26,112.
