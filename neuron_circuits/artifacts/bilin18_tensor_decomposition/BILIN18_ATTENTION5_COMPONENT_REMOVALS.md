## H971 — Transported-value removal reproduces attention5's causal interaction effect

Previous goal turn was progress: H970 exact source/head factorization suggested a transported interaction. H971 performs finite output removals for total, value, routing, cross, five source roles and each of9 heads. For every quartet/state, subtract a*n times the selected native component from attention5's final-token write and recompute downstream computation, preserving the native attention value bus. Each component has paired norm-matched Gaussian factorial edits, plus a shared identity arm. These edits remove projected output terms; they do not directly replace attention patterns or source values inside the module.

All24 locked quartets remain included. The sole competent cohort consists of four familiar original-format quartets. Results below describe that cohort and remain same-panel exploratory evidence, not independent generalization validation.

| Removed component | Mean intended-margin loss | Matched random RMS | KL ratio relative to full removal |
|---|---|---|---|
| Full interaction |0.3297|0.0538|0|
| Transported value interaction |0.3293|0.0642|0.0119|
| Routing interaction |-0.0054|0.0087|0.9979|
| Routing/value cross terms |0.0063|0.0031|0.9937|
| Adjective source |0.1133|0.0352|0.3558|
| Following period source |0.1861|0.0364|0.2309|

The KL ratio is summed KL(full-interaction-removal output||component-removal output), divided by summed KL(full-removal output||native output). Zero matches the complete removal; one is the no-edit baseline. Value-only removal leaves about1.2% of that baseline discrepancy and reproduces the task-margin loss. Mean native-reference KL is0.006655 for value-only versus0.006723 for total. This strengthens the transported-content interpretation beyond the previous algebraic projection share.

The period source has a larger margin effect than the adjective despite similar native-trajectory vector shares. Other suffix tokens, prefix and explicit not-source terms have little effect under this assay. This does not imply that not is unimportant: it can alter representations at adjective and period positions before attention5 reads them. Role/head removal terms are sums of actual source contributions and do not depend on zero-padding allocation of P/V factorization.

Head-level effects differ from native vector share rankings: head7 has the largest margin loss0.1227 (random RMS0.0266); head8 loss0.0510 (random0.0045); head1 loss0.0405 (random0.0099). Head4, the largest native vector projection contributor, gives margin loss0.0323 (random0.0246), though it has a larger distributional effect than this margin alone suggests. No single head reproduces the total intervention distribution (best individual-head KL ratio0.5226 for head7). Component effects should not be summed as if nonlinear downstream responses were additive.

Audits:5,282/5,282 whole-model forward examples; identity logit error2.24e-5; exact component-sum residual7.39e-15 relative; matched random norm error2.13e-7. Independent manual forward path agrees exactly on a nontrivial value-component edit. Full-removal margins replay H969 within8.32e-6. Aggregates independently recomputed and period token verified at all96 source positions. Inputs unchanged and outputs finite. No training or reserved-test access; natural-document pool26,112.

Limits: privileged four-state teacher components, potentially off-manifold output edits, onlyfour competent quartets, and failed alternate formats. These results identify causally relevant transport of a composed result; they do not yet identify the earlier computation or establish a reusable feature manifold.

Next: trace residual/attention/MLP interaction at adjective and following-period positions through layers0–5, accounting for learned scales, normalization, and attention5's current-value versus layer0 value-bus mixture. Prioritize where interaction is newly introduced, then test finite upstream interventions. H961 frozen context-bridge predictor evaluation remains pending.
