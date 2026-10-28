## H551 — MLP and attention responses alter the copying edge's downstream meaning

Posthoc follow-up to the H550 L8H3 copying failure. Retain all 16 copying prompts and all four nonnative position conditions. Run 256 interventions with four downstream modes: live, native MLP writes clamped at L8–L17, native attention writes clamped at L9–L17, and both. Clamps replace whole writes at every position, preserving the L8 attention edge intervention. These are broad dependency tests, not identification of individual neurons or a unique mediator.

| Downstream condition | Frozen-prediction direction agreement | Mean absolute effect | Prediction MAE |
|---|---:|---:|---:|
|live|42.2%|0.045203|0.098785|
|freeze_mlp|100.0%|0.055037|0.017299|
|freeze_attention|87.5%|0.057147|0.041317|
|freeze_both|100.0%|0.069637|0.000002|

Live computation agrees with the frozen prediction in 27/64 cases. Freezing MLP responses restores agreement in 64/64; freezing later attention restores 56/64. Freezing both exactly recovers the analytical final-residual prediction within 5.73e−6 nats. This numerical replay supports the intervention accounting rather than merely matching a direction label.

MLP responses are therefore important to the observed direction mismatch under these clamps; attention responses also matter. This is not an additive decomposition into independent MLP and attention shares: the components interact, and freezing either alters the inputs seen by the other. It does not establish that one particular MLP is necessary or that the same mechanism holds for all copying prompts.

The working unit of computation needs a source, a conditional reader, and its downstream consumers. A token-identity write may have a different live task effect than its frozen final-readout effect. H550 binding at these heads is more directly predictable; L8H3 copying requires a richer causal account. The original question remains unresolved at the level of a general ontology, but this supplies an experimentally testable distinction between direct readout and downstream transformation.

Next H552: localize the MLP response across layers using single-layer clamps and cumulative boundaries on the same panel, then freeze any selected mediator before testing new copying templates and binding controls. Preserve sign reversals and test attention interactions. Also retain physical-versus-virtual position validation as an outstanding branch; do not treat the virtual intervention as proof about natural word order.

Audits: native replay max error 7.63e−6; live H550 replay 2.39e−5; both-clamped prediction replay 5.73e−6; all finite; source unchanged. H550–H551 total 876 interventions, no new prompts or FineWeb documents, tracked corpus total 26,112. Broad goal active and incomplete.
