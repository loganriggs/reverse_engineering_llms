## H540 — A frozen signed-reader rule predicts new identity interventions and gate reversals

H539 found that the leading heads' identity maps were consistently negative, while routing sign determined copying versus suppression. Its secondary coefficient-conditioned rule is now tested prospectively. Keep the four heads and all identity-map signs from the original H539 discovery locked; do not refit on the new panel. The prediction is:

`sign(third-token contrast effect) = sign(locked identity-map sign * native routing coefficient * intervention gate)`.

The native coefficient is measured before intervention from that prompt's routing; it is an input to the explanation, not a learned prediction of how routes are constructed. This experiment therefore tests the reader's conditional operation, not a complete task algorithm.

### Design

Use another eight lexical pairs per task in three formats, giving 48 new pairs and 96 new prompts. Vary the third-token pair with the lexical item: eight name pairs for binding and eight non-name word pairs for copying. Some third-token words appeared in previous work in other roles; this is new prompt/role and combination testing, not a model-training holdout. All six native formats pass the competence gate. No prompt or pair is excluded.

For each head and prompt, replace its persistent-bus contribution from the designated source positions with a third token's value. Multiply that contribution by a gate of +1, −1 or 0. Only this source-to-head bus contribution changes, not the entire head's attention or query/key projections. Keep all downstream computation active. Compare the third0-minus-third1 output margin under third0 injection with that under third1 injection. A positive effect is relative identity copying and a negative effect is relative identity suppression. This does not assert that every individual token logit changes independently or that the full-vocabulary operator is exactly plus/minus identity.

Gate −1 tests reversal of the proposed operation. Gate 0 makes the two third-token interventions identical and provides an exact invariance control. Additional self interventions add zero attention change. The locked signs and complete manifest are saved before fresh geometry and causal evaluation.

### Prospective results

| Task | Reader | Gate +1 sign predictions | Gate −1 sign predictions | Mean absolute effect, +1 | Mean absolute effect, −1 |
|---|---|---:|---:|---:|---:|
|entity_binding|L14H4|48/48|48/48|3.1818|3.1351|
|entity_binding|L8H3|48/48|48/48|0.3007|0.3022|
|entity_binding|L14H0|48/48|48/48|0.1220|0.1218|
|entity_binding|L8H0|23/48|23/48|0.0006|0.0006|
|sequence_copy|L14H4|48/48|48/48|1.7414|1.7433|
|sequence_copy|L8H3|48/48|48/48|2.3228|2.2850|
|sequence_copy|L14H0|48/48|48/48|0.4394|0.4440|
|sequence_copy|L8H0|17/48|17/48|0.0035|0.0034|

The two leading heads predict all 384 nonzero-gate comparisons correctly across both tasks and both signs. The smallest leading-head effect is about 0.061 log-odds, well above observed self-control numerical error. Gate reversal reverses the identity effect as predicted despite downstream computation remaining active. All zero-gate comparisons agree within 3.6e-6.

The L14H0 control also follows its own locked signed-identity rule in 192/192 cases, with smaller mean effects. Its identity map is positive, unlike the two leaders, demonstrating that coherent identity readers are not unique to the selected heads. L8H0 predicts only 80/192 cases and has much smaller effects with unstable identity alignment. This control comparison supports an interpretable family of identity transformations; it does not establish that the leading heads are uniquely meaningful.

The 384 successes are structured comparisons across 96 prompts, with correlated lexical pairs, formats, directions and gate settings. They are not 384 independent linguistic samples or a population-accuracy guarantee. The intervention's supplied coefficient is measured on each prompt, so the result does not show that a task label alone determines the effect. Indeed H539 already falsified that simpler interpretation for some copying contexts.

### What this adds to the research objective

We now have a concrete conditional operation with prospective causal predictions: an identity-aligned or identity-opposing token transformation, multiplied by a signed contextual reader coefficient. The same map can produce copying or suppression when the coefficient changes sign. This supports read–transform–write relations as useful candidate units here, more directly than a fitted direction that merely steers familiar answer words. It describes a composite input/output relation rather than assigning meaning to one arbitrary internal coordinate.

It remains a partial mechanism. We have not explained how the native query/key computation chooses that sign, established the pathway's necessity in natural execution, shown that all source roles use the same operation, or derived a canonical ontology for the full model. The unit is tested on specific token-identity contrasts and hybrid source/head interventions. Other routes and MLP computations remain important, particularly for copying.

Hypothesis priorities: the source/reader account and reusable conditional identity transformations are now strongest for these two tasks. The earlier late-MLP answer-word-bias explanation remains relevant to H535; these findings do not rehabilitate those steering vectors as general features. The next plausible confound is lexical rather than relational routing: binding ends with “to,” whereas copying ends with a key token. The reader sign could partly follow that cue instead of the source's role.

Next H541: cross binding and repeated-copy contexts using the same names and the same final token (for example the copying key “to”), retaining the fixed heads and sign rules. Check native competence first. Compare routing signs and interventions while counterbalancing source order and positions. If the role-dependent sign persists with matched lexical cues, inspect the two query/key factors separately to locate where context supplies it; if not, narrow the interpretation to a lexical/context cue rather than a general role operation. These tasks directly distinguish hypotheses and should precede further compression work.

### Verification

2,784 causal interventions and 1,152 third-token comparisons are archived, including 384 zero-gate comparisons. Token-value replay error <=1.34e-5; coefficient factorization relative L2 error <=9.9e-5 (small/cancelling contributions make relative error larger); self margins agree within 5.8e-6. All measurements finite, source and manifest hashes unchanged. Independently verify the locked-sign file hash against the preregistered config. No new FineWeb documents; tracked total 26,112. This turn added 192 fresh synthetic prompts across H539/H540 and retained all declared cases. Broad goal remains active and incomplete.
