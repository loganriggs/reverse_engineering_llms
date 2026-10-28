## H578 — Contextual suffix values dominate the tested producer change

Align suffix tokens between base and edited prompts, excluding the final token. Assert exact token identity and equality of persistent bus vectors. Cross base/native attention patterns with base/native mixed values for L4H0 and L7H8. Only the suffix contribution is replaced; other sources and all later computation remain live. Mode names put routing first and values second: base_native restores routing only; native_base restores values only. Four combinations give 1,024 interventions on the reused copying panel.

| Condition/head | Both restored | Routing restored only | Values restored only |
|---|---:|---:|---:|
|copying_old_paren_L4H0|0.903|0.985|0.919|
|copying_old_paren_L7H8|0.847|0.991|0.858|
|copying_old_comma_L4H0|0.858|0.956|0.885|
|copying_old_comma_L7H8|0.806|0.963|0.851|
|copying_new_paren_L4H0|0.681|0.924|0.764|
|copying_new_paren_L7H8|0.950|0.997|0.957|
|copying_new_comma_L4H0|1.262|1.014|1.212|
|copying_new_comma_L7H8|0.936|1.011|0.932|

Restoring values generally reproduces substantially more of the suffix effect than restoring routing. For L4H0 in new-template parentheses, both restores give residual ratio 0.681, values-only 0.764 and routing-only 0.924. In new-template commas, values-only produces 1.212, close to both-restored 1.262, whereas routing-only is 1.014. Thus the sign-changing influence largely remains when routing is held native and only contextual values are changed.

The bus vectors match exactly at aligned suffix tokens. Consequently the value difference comes entirely from the local context-dependent value path, after its fixed mixture coefficient. An exact vector expansion into routing-only, value-only and interaction terms is retained in geometry. The value term's average projection onto the total suffix-write change ranges about 0.78–1.00 across cells, but these signed projections can cancel and are not independent causal percentages. The live crossings provide the stronger evidence.

This refines the mechanism: inserted text changes representations at later suffix tokens; producer heads read those contextualized values into the destination; this alters the subsequent L14 query and its name read. It is a partial causal chain, not an end-to-end circuit proof. We have not identified which earlier computation writes that information into suffix values, its semantic content, or whether the same carrier works on unrelated tasks.

Next H579 should localize the suffix contribution to individual aligned positions (yesterday, punctuation, pronoun and spoke/wrote in these templates), replacing only local values while holding routing/bus fixed. Preserve all punctuation/template conditions and full-suffix controls. Then trace the strongest carrier's input through earlier layers and test lexical/template transfer before assigning a feature label.

Audits: suffix tokens aligned; persistent bus difference exactly zero; base-suffix replay within 1.05e−5 margin; self replay <=2.53e−5; source unchanged, finite outputs. H577–H578 total 4,864 interventions, no new prompts or FineWeb documents; tracked total 26,112. Broad goal active and incomplete.
