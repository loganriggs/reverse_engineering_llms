## H537 — Most bus-induced answer change survives fixed downstream readers

The previous turn made progress: H535 weakened the general-feature interpretation of selected late MLP writes, and H536 localized larger binding/copying effects to attention and the persistent token-value bus. This experiment keeps H536's bus-injection layers fixed (binding after L6, copying after L3) and reuses every discovery/transfer pair. No layer or pair is reselected.

For each donor-bus substitution, compare five conditions: unrestricted downstream computation; recipient-native Q/K projections fixed; recipient-native contextual V projections fixed; both Q/K and V fixed; and Q/K, V plus MLP outputs fixed. Freeze all positions in all layers strictly after bus injection. Both Q/K pairs are fixed before their native normalization and rotation, so the complete squared-attention pattern stays native. Contextual V is the local value projection; the injected persistent bus remains changed. The final condition also fixes the entire native MLP output, allowing only direct attention bus contributions, residual mixing and the final nonlinear readout to respond.

Include task donors in both directions, base-to-nuisance donors, and self donors. Six nuisance pairs have unequal token lengths and are excluded from all five conditions; no task pair is excluded. The same recipient-native projections are saved in memory before intervention, and hook counts are asserted at runtime.

| Family | Condition | Transfer donor-directed log-odds change | Ratio to unrestricted bus effect | Mean absolute nuisance effect |
|---|---|---:|---:|---:|
|entity_binding|free|7.5035|100.0%|0.0350|
|entity_binding|fixed_qk|7.3207|97.6%|0.0359|
|entity_binding|fixed_v|7.4374|99.1%|0.0395|
|entity_binding|fixed_qkv|7.2446|96.5%|0.0291|
|entity_binding|fixed_qkv_mlp|7.0035|93.3%|0.0262|
|sequence_copy|free|6.8117|100.0%|0.7683|
|sequence_copy|fixed_qk|6.7925|99.7%|0.6307|
|sequence_copy|fixed_v|6.2842|92.3%|0.4681|
|sequence_copy|fixed_qkv|6.2926|92.4%|0.3119|
|sequence_copy|fixed_qkv_mlp|4.5611|67.0%|0.2584|

All 16 transfer task directions remain positive in every condition. Fixing Q/K alone retains 97.6% of binding and 99.7% of copying's unrestricted bus effect. Fixing Q/K and contextual V together retains 96.5% and 92.4%. Fixing MLP outputs as well retains 93.3% and 67.0%. Thus a large direct route from changed token values through recipient-native readers to the answer is sufficient in these interventions. Copying has more additional effect from downstream MLP response than binding under this particular conditional comparison.

These percentages compare mean answer log-odds changes, not task accuracy or unique causal mediation fractions. They are conditional on donor-bus intervention and on original recipient routing; they do not establish that native Q/K computations are unnecessary. Those computations supply the readers being held fixed. Nor is the additional effect of unfreezing a component a unique additive share: interactions among routing, values and MLP responses remain possible. The large intervention changes a stream available to many downstream layers.

The key update to H533 alternatives is that *reading token values through existing context-dependent routes* is more directly supported than *needing to change those routes* for these paired perturbations. This favors an explanation in terms of a source, a reader and its output transformation. It is compatible with learned features elsewhere and does not imply a complete native algorithm has been identified.

Next: derive the fixed-QKV/MLP response as an explicit sum of layer/head/source contributions, validate its final hidden state and margins against the actual intervention, then rank layer/head readers using discovery only and freeze them for transfer. Exact replay of the frozen system is expected from its algebra; the informative tests are its agreement with the unrestricted intervention and whether specific readers transfer across task variations.

Audits: 610 interventions; self margins agree within 3.9e-6 and self KL within 2.2e-7; unrestricted replay agrees with H536 within 1.3e-5. All freeze-hook counts asserted, outputs finite, source and manifest hashes unchanged. New numerical comparisons reflect batch/kernel rounding. No new prompts or FineWeb documents; tracked FineWeb total 26,112. Full-vocabulary KL and all contrast measurements are archived. These remain small, deliberately constructed task panels, not population estimates.
