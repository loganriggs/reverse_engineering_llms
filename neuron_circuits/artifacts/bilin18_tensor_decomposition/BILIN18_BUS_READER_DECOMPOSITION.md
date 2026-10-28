## H538 — An explicit token-to-reader computation reproduces the frozen bus intervention

H537 showed that a large fraction of the donor-bus effect survives when downstream routing, contextual values and MLP outputs are fixed to their native recipient states. Here derive and execute that conditional computation. Keep the H536/H537 injection layers and the full H533 discovery/transfer panels unchanged.

For the final query position, each downstream attention layer/head reads a weighted sum of the changed initial value-bus vectors at source positions. Apply that head's output projection, its learned persistent-bus coefficient, and the product of all subsequent residual mixing coefficients. Sum across all downstream layers and heads, add to the native final residual state, then apply the native final normalization, unembedding and logit softcap. In symbols:

`delta_final = sum_l,h,s [residual_gain_after_l * bus_mix_l * native_pattern_l,h(final,s) * output_projection_l,h * delta_bus_s,h]`.

The pattern is the model's actual product of two normalized query/key score factors; it is not a softmax distribution and can be signed. Native fixed contextual values and MLP outputs cancel in the difference. No predicted semantic label is supplied to this formula. Its coefficients do come from the full recipient-native forward pass, so this is a conditional model of an intervention, not a standalone replacement for the native model or an explanation of how those coefficients were constructed.

### Replay and source localization

Across all 64 task directions, the formula's answer margins match actual fixed-QKV/MLP intervention margins within 3.9e-6. The maximum final-hidden error is 0.0129 in absolute units but only 2.9e-6 relative to the intervention's hidden-state change; accumulated activations have large scale. Actual intervention margins also replay H537 within 3.9e-6. These are float32 numerical differences with the same bfloat16-rounded weights.

Every binding pair changes exactly one token position: the repeated giver name. Every copying pair changes exactly two positions: the repeated value token. The persistent bus difference is exactly zero at all unchanged positions. Thus the response formula localizes the manipulated source information as well as its downstream readers. This is more specific than calling an output direction a “binding feature,” but it does not yet prove whether the readers suppress the giver, select a recipient, or implement another transformation.

Exact replay is expected once the model is frozen this way and validates the implementation; it is not itself evidence of a discovered natural mechanism. The informative link is H537's result that the frozen response retains about 93% of unrestricted binding bus effect and 67% of copying bus effect.

### Reader selection and transfer

Treat each layer/head pair as a reader, summing its contributions across source positions. Measure the finite answer-contrast effect of each reader alone. Rank readers by mean donor-directed effect on discovery examples, freeze the ranking, and evaluate summed top-1/3/5/10 reader contributions on every transfer pair. Compare with 32 seeded uniformly chosen reader sets at each size. These random controls match reader count, not weight magnitude, spatial distribution or training history; they measure concentration, not uniquely semantic specialization.

| Family | Selected readers | Transfer contrast change | Fraction of all fixed readers | Range of 32 random-set mean effects |
|---|---:|---:|---:|---|
|entity_binding|1|3.0264|43.2%|[-0.0900, 0.5714]|
|entity_binding|3|4.2182|60.2%|[-0.1292, 0.5708]|
|entity_binding|5|5.2325|74.7%|[-0.0996, 3.2099]|
|entity_binding|10|6.6292|94.7%|[-0.1039, 4.2179]|
|sequence_copy|1|1.4340|31.4%|[-0.0243, 0.4780]|
|sequence_copy|3|1.8917|41.5%|[-0.0897, 1.4258]|
|sequence_copy|5|2.7557|60.4%|[-0.1037, 1.4377]|
|sequence_copy|10|3.8085|83.5%|[-0.2270, 1.6806]|

Binding's top reader is L14 head 4; its next two are L13 heads 5 and 0. Copying's top reader is L8 head 3; its next two are L14 head 4 and L8 head 4. Thus the same head can contribute to multiple tasks: L14 head 4 is first for binding and second for copying, while L8 head 3 is first for copying and sixth for binding. Layer/head numbers are zero-based. This overlap is consistent with reusable token transformations whose contextual routing determines their task role; it is not proof of a single shared semantic operation.

The top ten readers carry 94.7% of the fixed-reader binding effect and 83.5% of the copying effect on transfer. The best reader alone carries 43.2% and 31.4%. Selected sets exceed all 32 random-set mean effects at each tested size. All task directions are positive except one direction for copying's top-three set. Finite readout effects do not add linearly: selected-reader effects are computed after summing their residual contributions and applying final normalization and softcap. Fractions are relative to the complete frozen-reader contrast, not native accuracy or exclusive circuit shares.

### Hypothesis update and next falsification

The most concrete current candidate unit is a relation between a token-value source, a context-dependent signed attention coefficient, and a head-specific output transformation. H535 showed why a late task-steering vector alone can be misleading. H537/H538 instead provide an executable, source-specific causal account for a restricted intervention, with transferable reader choices. We still have not explained the native creation of the routes, established necessity, or shown that these readers retain the same operation across broader tasks.

Next test: separate the sign of the native attention coefficient from the token-identity transformation in the leading shared heads. Determine whether L14 head 4 copies or suppresses the giver identity in binding, and how its operation changes when reading repeated values in copying. Make those predictions using discovery only, then test fresh names, values, and role/position changes. Include matched reader/location controls and counterfactuals that distinguish suppressing a mentioned entity from selecting the intended recipient. Do not promote “signed token transport” to a general ontology until those tests succeed.

Artifacts retain all 7,200 individual-reader measurements, 4,256 selected/random-set transfer evaluations, 64 actual intervention replays, source-position records and final-residual contribution vectors. Source and manifest hashes are unchanged, all values finite, and zero bus change at unchanged positions is verified. No new prompts or FineWeb documents; tracked total 26,112. The broad goal remains active and incomplete.
