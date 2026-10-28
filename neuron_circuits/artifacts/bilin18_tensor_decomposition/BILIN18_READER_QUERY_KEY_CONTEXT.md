## H542 — Context at both query and key selects the copying sign

Reuse H541's exactly matched standard-binding and position-matched-copying prompts: eight name pairs, two source identities per pair. Selected source and final tokens, their absolute positions, and the third-token alternatives match across contexts. Fix the same four heads and the H539 identity-map signs. No new heads or semantic directions are fitted.

Cross the origin of query representations (binding B or copying C) with the origin of key representations (B or C). Replace both actual query projections for the chosen head at the final position, and both actual key projections at the matched source position. Normalization and rotation then run natively. Other projections remain native and all downstream computation runs normally. These are actual Q/K swaps, not merely changes to an effective coefficient.

Query replacement can affect other source edges of that head; key replacement can affect other query destinations. The experiment therefore does not isolate one attention edge's total behavioral effect. To test the reader's conditional identity operation within each recombined state, additionally replace the designated edge's persistent-bus contribution with each third-token alternative, using the recombined coefficient. The difference between their output contrasts tests copying versus suppression with the altered attention state and all downstream response active.

### Context factorial

The table reports coefficients and effects for the two leading heads. Coefficient counts use the 16 matched name/direction cases; causal third-token results include both recipient contexts, giving 32 cases per row. Positive identity effects copy the injected identity; negative effects suppress it relatively.

| Reader | Query origin | Key origin | Mean coefficient | Positive coefficients | Mean third-token identity effect | Frozen sign prediction |
|---|---|---|---:|---:|---:|---:|
|L14H4|B|B|1.4791|16/16|-3.0179|32/32|
|L14H4|B|C|0.5754|16/16|-1.1764|32/32|
|L14H4|C|B|0.4014|16/16|-0.8144|32/32|
|L14H4|C|C|-0.0477|5/16|0.1008|32/32|
|L8H3|B|B|0.1405|16/16|-0.3005|32/32|
|L8H3|B|C|0.0322|16/16|-0.0690|32/32|
|L8H3|C|B|0.1084|16/16|-0.2339|32/32|
|L8H3|C|C|-0.1402|0/16|0.3008|32/32|

L8H3 switches to the copying sign only for C-query/C-key: all sixteen such coefficients are negative, while all three other combinations are positive. Transplanting either copying operand alone into the binding combination is insufficient to reverse the sign. The conditional identity operation follows the resulting sign in every tested recipient context. This locates relevant contextual distinctions on both sides of the edge, rather than solely in the query or solely in the key.

L14H4 likewise has positive coefficients in all mixed and B/B combinations, but C/C is weaker and mixed (eleven negative, five positive), consistent with H541. Do not generalize its role to unconditional copying. Both leading heads keep their second Q/K dot-product factor positive in every factorial case; the first factor supplies the sign changes.

The observed 2×2 sign table does not prove a discrete logical AND gate or that the bilinear cross term is uniquely necessary. Continuous changes in dot products, including additive changes crossing zero, can produce the same table. A secondary algebraic check illustrates the distinction: for L8H3, the first-factor additive extrapolation from B/B plus the two single-context changes is already negative in 14/16 cases, though the exact cross term makes it more negative on average. The valid primary claim is the actual counterfactual sign behavior of the four tested input combinations.

Across leading heads, the frozen identity rule predicts 256/256 third-token effects under actual Q/K recombination. L14H0 also predicts 128/128 with smaller effects, while L8H0 predicts 80/128 and remains weak/inconsistent. These repeated structured cases are not independent population samples. Correct signs do not imply exact prediction of effect magnitude or complete task accuracy.

### Research update

Together, H541/H542 rule out final-token identity and matched absolute positions as sufficient explanations for the leading L8H3 sign switch, and show that context-derived representations at both the query and key participate. They strengthen a conditional read–transform–write account: contextual query/key factors select a signed reader coefficient, and a relatively stable identity transformation determines what that reader does to token values.

They do not yet establish abstract “giver” or “recipient” features. Narrative versus list format, local surrounding words and contextual states still covary with the intended operation. Actual Q/K transplants are hybrid interventions, and upstream construction of those states remains unexplained. The broad objective is therefore incomplete.

Next H543: cross surface format and intended operation. Keep names and final “to” shared, add narrative copying (repeated sentences ending in “sent to A”), and list-form binding/retrieval (for example “from A to B; from B to A; from A to”). Retain narrative binding and list copying anchors. Freeze the heads and sign rule, screen native competence, and report all outcomes. This separates a role/operation-linked gate from a prose/list heuristic; local source cues and positional distances should be recorded and counterbalanced where feasible. Failed competence must not be treated as mechanistic failure. Only then consider transporting a learned contextual feature or pursuing source/key construction upstream.

### Verification

16 matched token/position records, 256 routing-factor records, 1,536 interventions and 512 causal third-token comparisons. All projection-hook counts asserted; self Q/K swaps reproduce native margins within 1.1e-5. Native B/B and C/C coefficients replay H541 within 2.1e-7. The matching audit asserts equal source/final token IDs, source/final positions and third-token sets. Outputs finite and both source JSON and native-cache hashes unchanged. No new prompts or FineWeb documents in this diagnostic; H541 added 80 new copying prompts earlier this turn. Tracked FineWeb total remains 26,112.
