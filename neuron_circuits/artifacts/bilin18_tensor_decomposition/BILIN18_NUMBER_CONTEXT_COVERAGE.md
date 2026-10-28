## H596 — Broader discovery coverage improves response capacity at fixed rank, but fidelity objectives still disagree

Previous turn made progress by separating coefficient-objective error from the attainable local-response span. Now compare old32 (fit on regular/fronted contexts), new32 (top 32 directions fit on regular/fronted plus regular/relative-clause contexts), and new64 (all 64 directions from the expanded set). Old32 versus new32 holds dimension fixed; new32 versus new64 holds fitting data fixed. Expanded fitting uses eight regular lexical pairs x four subject/attractor-number states x two frames =64 context differences, not 64 independent nouns. No irregular nouns enter fitting.

Evaluate every irregular fronted and relative-clause prompt at both signed doses. For each span compare ordinary context projection and a per-case target-informed response-optimal oracle. Full context-switch and unchanged-context references remain. Total 2,048 live interventions. Oracle coefficients optimize local Euclidean response only, not downstream KL or margin, and are not learned predictions. All prompts have been explored in prior experiments.

| Small-dose irregular cell | Span / coefficients | Local response error | Relative margin error | Relative reference KL |
|---|---|---:|---:|---:|
|fronted_preposition|old32_projection|0.6020|0.6355|0.5103|
|fronted_preposition|old32_oracle|0.5523|0.1431|0.1616|
|fronted_preposition|new32_projection|0.5909|0.5200|0.3425|
|fronted_preposition|new32_oracle|0.5524|0.1687|0.1540|
|fronted_preposition|new64_projection|0.5330|0.3676|0.1964|
|fronted_preposition|new64_oracle|0.4927|0.1201|0.1146|
|relative_clause|old32_projection|0.8824|1.1265|0.7646|
|relative_clause|old32_oracle|0.8431|0.3100|0.3023|
|relative_clause|new32_projection|0.4490|1.1507|0.3256|
|relative_clause|new32_oracle|0.4060|0.2091|0.0658|
|relative_clause|new64_projection|0.3904|0.5966|0.1210|
|relative_clause|new64_oracle|0.3593|0.1626|0.0476|

Coverage makes a large difference for relative clauses at fixed dimension: oracle local-response error falls 0.843 to 0.406, relative reference KL 0.302 to 0.066, and relative margin error 0.310 to 0.209. Thus the previous failure cannot be attributed simply to a requirement for more than 32 dimensions. Which contexts define those dimensions matters.

Ordinary projection shows the metric distinction sharply: local error improves 0.882 to 0.449 and relative KL 0.765 to 0.326, but relative margin error does not improve (1.126 to 1.151). Better vector reconstruction or distributional fidelity is not a guarantee of better agreement-contrast fidelity.

The fronted cell does not improve uniformly at fixed rank. Oracle local error stays near 0.552 and margin error slightly worsens (0.143 to 0.169), although KL improves slightly. Redistributing a fixed-dimensional basis toward another frame has a tradeoff. Increasing the expanded basis from 32 to 64 improves both cells: new64 oracle local errors are 0.493 fronted and 0.359 relative clause, with relative KL 0.115 and 0.048. The half-dose pattern is similar. Substantial internal error remains; these results are not a proof of a general compact context representation.

New32 retains 98.7% of expanded discovery context energy, again showing that energy alone does not settle causal fidelity. The new64 span contains the old span to numerical tolerance, and its response oracle is never worse in the optimized local metric. That monotonicity is not asserted for KL or answer margin. The maximum response-matrix condition number is 11.69, so the improvement is not explained by an unstable inverse.

Next freeze the basis choices, number direction, dose and evaluation criteria, then test new lexical combinations and wording before further tuning. Keep same-wording fresh-lexicon controls to distinguish lexical from frame transfer. Recheck native task competence without filtering failures. Compare the full causal context switch, ordinary projection and oracle; if broader coverage transfers, then pursue a target-free coefficient model. An oracle's success alone is not evidence that such coefficients can be predicted or interpreted as a general feature.

Audits: 64 regular context differences, zero irregular fitting rows; old-span containment relative error 1.82e-6; normal-equation residual <=7.71e-15; coefficient replay <=5.83e-7; full/unchanged replay <=1.44e-5; old32 replay against H595 <=3e-5; nested-span oracle monotonicity verified. Source/bases unchanged, finite float64 KL/group-NLL metrics. No new prompts or natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
