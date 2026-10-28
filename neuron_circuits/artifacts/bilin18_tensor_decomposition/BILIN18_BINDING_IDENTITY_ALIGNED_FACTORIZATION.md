## H927 — Name-identity alignment removes most apparent routing–value interaction

The previous experiment found a large explicit cross term in a position-aligned finite-change decomposition. H927 tests whether that term persists when the swapped event names are aligned by identity. This directly addresses the user's data-based canonical-form idea: an explanation should distinguish stable computation from arbitrary correspondence between coordinates.

### Alignment and intervention

At the two event-name positions, jointly permute donor routing coefficients and donor mixed-value vectors so each name matches its recipient occurrence. Leave every other position unchanged. Both experimental contrasts reverse the event-name order, and token-ID equality is asserted after alignment. Joint permutation preserves the donor attention sum exactly; it is a reindexing of contributions after the model has run, not a modification to its positional encoding or evidence that the model performs this permutation internally.

With aligned donor coefficients and vectors, repeat routing-only, value-only, cross-only, simultaneous routing+value, all-three, cached full replacement and identity interventions. The same three exploratory heads, all128 familiar transfer rows and both contrasts are retained.

### Causal comparison

Cross effects are RMS changes in the second-event-name margin. Routing error compares the aligned routing-only intervention with the unchanged full donor-head replacement, normalized by the full effect's RMS. This is answer-margin fidelity, not yet full-distribution fidelity to the full edited output.

|Contrast|Head|Position-aligned cross RMS|Identity-aligned cross RMS|Aligned routing vs full RMSE|Relative routing error|
|---|---|---:|---:|---:|---:|
|event_swap|L14H4|0.3817|0.00381|0.00302|1.22%|
|event_swap|L14H6|0.1969|0.00480|0.00379|3.68%|
|event_swap|L16H3|0.1783|0.00077|0.00114|1.42%|
|same_fact_reword|L14H4|0.3937|0.00404|0.00390|1.60%|
|same_fact_reword|L14H6|0.2009|0.00586|0.00569|5.63%|
|same_fact_reword|L16H3|0.1800|0.00142|0.00479|6.02%|

At L14H4, the event-swap cross term shrinks about100-fold, from0.38166 to0.00381. Aligned routing alone reproduces the full margin effect with1.22% relative error; same-fact rewording gives1.60%. All three heads show substantial reductions, though the weaker L16H3 rewording effect has6.02% relative routing error. Value differences at non-name positions remain for rewording, so name alignment is not expected to erase all contextual change.

The full swap's effect remains exactly the same as in the earlier experiment. Its apparent attribution changes: before alignment, moved lexical vectors created large value and cross terms; after alignment, the difference is mostly reassignment of coefficients over stable lexical identities. These two decompositions describe the same total head-output change but define different intermediate interventions. Their components should not be mistaken for unique, invariant causal units.

### Research implication

This is stronger evidence for identity-indexed routing over lexical carriers than for a large standalone interaction feature in this panel. It also qualifies H926: the large cross term was necessary for that position-aligned expansion, but was not itself evidence of an independently meaningful computation unit. Small lexical spans failed, yet respecting the correspondence between identities produces a simple and faithful answer-margin account.

The alignment uses known token identities and event occurrences. It is an oracle correspondence, not a learned unsupervised featurizer or a general canonical form. Repeated tokens, alternate tokenizations, contextual values and out-of-distribution contexts remain unresolved. The task itself still exhibits a shortcut; faithful modeling of that shortcut does not establish semantic binding.

### Next test

Evaluate a task-blind identity aggregation of the persistent-value pathway. Because its value vector is fixed by token identity, repeated occurrences of a token can be grouped by summing their signed routing coefficients, with that token's value decoded once. Verify this algebra and numerical fidelity across the existing agreement, language, binding and copy panels, then assess the remaining current-value pathway separately. Preserve the distinction between preserving native outputs and preserving position-specific interventions: merging occurrences loses access to individual-source edits unless provenance is retained.

For the approximate aligned-routing account, measure KL to the full edited distribution, not only the answer margin. Then test coefficient transfer across discovery/transfer identities and templates with the alignment rule fixed before outcomes. These steps would tell us whether a useful data-based representation extends beyond this hand-aligned name-swap panel, without prematurely declaring a manifold verified.

### Verification

5,504 forwards. Identity replay exactly zero; full-swap effects replay H925 exactly. Joint permutation changes the full attention sum by at most2.84e-14. Local factorization error6.22e-15; all-three terms reproduce full donor replacement within9.06e-6 maximum full-vocabulary logit error. Native-cache error2.05e-5; source hashes unchanged; all results finite. No reserved test or added natural documents; pool remains26,112. Broad goal active.
