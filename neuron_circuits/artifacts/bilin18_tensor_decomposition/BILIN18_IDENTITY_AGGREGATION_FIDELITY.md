## H928 — Identity-aligned routing preserves the full edited distribution; token aggregation is numerically exact

H927 showed low answer-margin error after matching swapped names by identity. H928 checks whether that conclusion survives full-vocabulary evaluation and audits a task-blind representation of the persistent-value pathway across diverse existing tasks.

### Full-distribution fidelity

For each of128 familiar QA transfer rows, both contrasts and each of three exploratory heads, compare full donor-head replacement with identity-aligned routing-only replacement and position-aligned routing-only replacement. KL is KL(full edited distribution || approximate edited distribution). The baseline is KL(full edited distribution || native distribution), corresponding to making no edit. Lower ratios are better. Top1 agreement is with the full intervention, not correctness on the task.

|Contrast|Head|Aligned KL|Aligned KL / no-edit|Position KL / no-edit|Top1 matches full|
|---|---|---:|---:|---:|---:|
|event_swap|L14H4|2.25e-06|0.00073|1.693|100.00%|
|same_fact_reword|L14H4|3.96e-06|0.00130|1.711|99.22%|
|event_swap|L14H6|4.4e-06|0.00732|1.801|100.00%|
|same_fact_reword|L14H6|7.51e-06|0.01236|1.780|100.00%|
|event_swap|L16H3|2.29e-07|0.00063|1.850|100.00%|
|same_fact_reword|L16H3|1.88e-05|0.05075|1.859|100.00%|

At L14H4, identity-aligned event-swap routing has KL2.25e-6 and retains only0.073% of the no-edit baseline discrepancy. Position-aligned routing is worse than no edit, with ratio1.69. Across all six head/contrast cells, aligned ratios range0.00063–0.05075, whereas position-aligned ratios range1.69–1.86. Thus answer-margin fidelity was not hiding a large overall distribution mismatch on this panel.

Fidelity is approximate, not exact: maximum vocabulary-logit errors range about0.0077–0.0867, and L14H4 rewording disagrees with full-edit top1 in one of128 rows. The weakest relative KL result is L16H3 rewording. These are comparisons on previously explored names, heads and templates, not independent broad generalization evidence.

### Task-blind token aggregation audit

The persistent value at each source position depends on token identity. Its attention contribution can therefore be written as a sum over unique tokens: each token's value vector multiplied by the sum of signed routing coefficients over its occurrences. The existing value-mixing scalar multiplies both sides. No task labels, semantic roles, clustering or learned sparse dictionary are needed for this equality.

A shared token/head dictionary takes the first observed vector for each token; subsequent occurrences are checked against it. Store source-position lists with each token so individual-occurrence provenance is retained. Audit the three previously captured readers L8H3, L13H0 and L14H4 across agreement, language, binding and matched copying,192 examples total.

|Domain|Family|Head/examples|Mean positions|Mean unique tokens|
|---|---|---:|---:|---:|
|agreement|task|96|5.00|5.00|
|agreement|copy|96|7.00|4.00|
|language|task|96|6.50|6.50|
|language|copy|96|7.00|4.00|
|binding|task|96|12.00|11.00|
|binding|copy|96|7.00|4.00|

All576 local component reconstructions agree to a maximum relative norm error2.34e-7 and absolute component error7.45e-6. Same-token values vary by at most1.53e-5 in the cached floating-point calculations. The audit spans discovery and transfer examples; dictionary extraction here is an algebraic validation, not a train/test feature-recovery experiment.

This is an architecture-derived identity, not an empirical discovery of a semantic manifold. It exactly reorganizes the persistent component and merges repeated tokens; it does not compress the full model, merge different tokens, explain routing, or replace current-layer values. Position-specific interventions still need the retained occurrence coefficients/provenance. No full-model forwards are claimed for the aggregation audit itself.

### Research implication and next test

A useful partial data-based representation is now concrete: reusable token-value vectors plus context-dependent signed coefficients, with positional provenance. It handles shared directions without counting repeated contributions incorrectly and makes the correspondence between swapped identities explicit. This is a structured baseline against which more flexible learned dictionaries, groups and DAGs should be judged; it does not establish their superiority or replace their manifold-validation requirements.

The remaining hard problem is predicting and explaining the coefficients. Next test whether discovery-fitted simple coefficient models preserve native outputs across agreement, language, binding and copying, using fixed template/position rules versus models conditioned on task context. Retain separate native-distribution KL and intervention fidelity: fitting outputs alone can hide the wrong causal response. Do not use oracle donor coefficients as evidence that the coefficient computation has been learned. Where such simple rules fail, localize which coefficient changes require contextual activations before choosing a richer featurizer.

### Verification

2,432 full-model forwards for edited-distribution comparisons;576 separate cached component aggregation audits. Native-cache max logit discrepancy2.05e-5; joint alignment permutation invariance error2.84e-14. Source hashes unchanged and all metrics finite. Token dictionary, occurrence provenance, all aggregation errors and all full-distribution comparisons saved. No reserved test or added natural documents; pool remains26,112. Broad goal active.
