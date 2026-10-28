## H552 — Copying response is distributed across MLP layers

Follow up H551 on the same 16 copying prompts and four virtual positions. Run all ten single-layer native MLP-write clamps at L8–L17, cumulative prefixes L8 through L8–L16, cumulative suffixes L9–L17 through L17, and four H551 controls. All other computation remains live unless specified. These 32 modes give 2,048 interventions. Select a single layer by minimum frozen-readout prediction MAE, with the selection rule fixed before evaluation; this panel is diagnostic discovery, not independent validation.

| Clamp | Direction agreement | Prediction MAE |
|---|---:|---:|
|live|42.2%|0.09879|
|freeze_mlp|100.0%|0.01730|
|freeze_attention|87.5%|0.04132|
|freeze_both|100.0%|0.00000|
|single_8|68.8%|0.05670|
|single_9|43.8%|0.06993|
|single_10|43.8%|0.08445|
|single_11|40.6%|0.09216|
|single_12|37.5%|0.09292|
|single_13|43.8%|0.09047|
|single_14|34.4%|0.09523|
|single_15|32.8%|0.09560|
|single_16|28.1%|0.10535|
|single_17|51.6%|0.06669|
|prefix_8|68.8%|0.05670|
|prefix_9|79.7%|0.04988|
|prefix_10|79.7%|0.04746|
|prefix_11|84.4%|0.03854|
|prefix_12|89.1%|0.03235|
|prefix_13|93.8%|0.02078|
|prefix_14|100.0%|0.01781|
|prefix_15|100.0%|0.01453|
|prefix_16|100.0%|0.01162|
|suffix_9|87.5%|0.03430|
|suffix_10|57.8%|0.04885|
|suffix_11|56.2%|0.05636|
|suffix_12|56.2%|0.05868|
|suffix_13|56.2%|0.06001|
|suffix_14|57.8%|0.06459|
|suffix_15|57.8%|0.06125|
|suffix_16|56.2%|0.06334|
|suffix_17|51.6%|0.06669|

L8 is the best single-layer clamp under the fixed selection rule: MAE falls from 0.09879 to 0.05670 nats and sign agreement rises from 27/64 to 44/64. L17 and L9 also reduce MAE substantially. No single-layer clamp restores complete direction agreement. Freezing MLPs cumulatively from L8 through L14 does restore 64/64; L8–L16 yields smaller MAE than freezing all MLPs, so late responses can oppose earlier responses rather than merely adding errors. Prefix L8 and suffix L17 duplicate their single-layer controls intentionally.

This supports a distributed downstream transformation, with a strong same-layer response, not a single uniquely identified MLP mediator. MLP clamping also changes subsequent attention inputs; effects cannot be summed as independent layer contributions. Whole-write clamps do not identify individual features or neurons.

Next H553 fixes L8 before evaluating new prose-code and record-list copying templates plus binding controls. Compare live, L8-only, all downstream MLPs, and both attention/MLP clamps. Require native competence reporting and retain all cases. The prediction is that the selected L8 clamp reduces frozen-readout error on the new copying templates; broad-clamp success alone would not validate a compact mediator.

Audits: H550 live replay within 2.39e−5, native replay within 7.63e−6, both-clamped analytic replay within 5.73e−6; all finite and source unchanged. No new prompts or FineWeb documents. Tracked total 26,112. Broad goal active and incomplete.
