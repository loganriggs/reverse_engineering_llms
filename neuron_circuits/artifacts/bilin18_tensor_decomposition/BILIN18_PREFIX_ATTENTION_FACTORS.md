## H612 — Attention5 interaction is carried by contextual values with coordinated routing/value effects

H611 localized an influential Attention5 body write. Decompose it exactly before attributing the effect. Reconstruct native signed attention patterns from RMS-normalized rotary Q/K and Q2/K2. Separate the contextual local values from persistent values using the native coefficients: local weight 1.056396484375 and persistent weight -0.056396484375. These are signed learned mixing coefficients, not probabilities.

Align body query positions and pad absent prefix-key slots with zero routing and zero values in unprefixed runs. This preserves the exact output; it also makes the expansion convention explicit. For each value stream write the output as F(R,V), including the output projection, and anchor the four-condition expansion at unprefixed uppercase:

- Routing-only term: F(R_interaction,V00).
- Value-only term: F(R00,V_interaction).
- Mixed term: the full interaction minus those two terms.

The mixed term collects all remaining co-variation, including appearance of prefix slots. This is an exact baseline-anchored decomposition, not a unique attribution or a claim that all three terms are independently meaningful features.

Native reconstruction passes before interventions. On combined prefix/lowercase prompts subtract full, local, persistent, routing, value, mixed, and stream-specific components across the body, with per-position norm-matched random controls and identity. 800 interventions; same panel and no new fitting. Persistent values remain live during these output-write subtractions.

**The direct Attention5 interaction is overwhelmingly carried by the contextual-value stream.** Its norm is about 1.001–1.002 times the full interaction, while the persistent-stream contribution is about 0.041–0.044. Removing the contextual stream leaves target MLP11 interaction norm 0.577/0.509, almost matching full removal at 0.574/0.507. Removing the persistent-stream contribution leaves 0.997/1.000. This concerns the direct Attention5 output contribution; persistent values may still affect earlier computations that produce its routing and local values.

| Component removed | Old target norm | Old random control | New target norm | New random control |
|---|---:|---:|---:|---:|
|full|0.5742|1.1389|0.5070|1.1154|
|local|0.5772|1.1394|0.5087|1.1159|
|bus|0.9968|1.0031|0.9997|1.0001|
|routing|0.9422|1.2310|0.9774|1.1611|
|value|0.7087|1.2670|0.9560|1.2990|
|mixed|1.2309|1.3439|1.1769|1.4045|
|local_routing|0.9412|1.2308|0.9753|1.1613|
|local_value|0.7087|1.2670|0.9560|1.2990|
|local_mixed|1.2334|1.3422|1.1811|1.4031|
|bus_routing|1.0009|1.0012|1.0032|1.0006|
|bus_value|1.0000|1.0000|1.0000|1.0000|
|bus_mixed|0.9959|1.0030|0.9966|0.9999|

The persistent values themselves have exactly zero four-condition interaction after token alignment. Their output interaction can still be nonzero because routing changes and combines with separate prefix/case value changes. Zero value interaction therefore does not mean zero pathway contribution.

Within the contextual stream, the components are large and cancel. Mean routing/value/mixed norms relative to the full interaction are about 1.17/1.38/1.54 in old wording and 1.28/1.34/1.65 in new wording. Removing mixed terms actually increases the later target norm to 1.231/1.177; routing-only removal leaves 0.942/0.977, and value-only removal leaves 0.709/0.956. Full removal is more effective than any single factor, particularly in new wording. These downstream effects cannot be summed: component cancellation, nonlinear propagation and norm rotation all matter.

Every tested intervention retains correct full-vocabulary copying top1. Mean absolute output KL for full removal is 0.00461/0.00399; contextual-stream removal gives 0.00437/0.00388; persistent removal gives 0.000013/0.000019. These are absolute changes from the native combined prompt, not relative fidelity errors or shares of task performance. Internal target norms, probability changes and task success remain different outcomes.

Interpretation: the localized object is a coordinated routing-and-contextual-value computation. It is not well described here as a single preexisting value feature, and it is not primarily a direct persistent-value read. The factorization does not yet identify a minimal or unique unit; its baseline convention and the restricted prompt family matter. The result nevertheless narrows the search from an entire layer/bus system to Attention5's contextual-value interaction across body positions.

Next resolve this interaction by attention head, retaining exact output additivity and testing individual removals and leave-one-head-out reconstructions. This can distinguish a compact head-level carrier from distributed cancellation. Compare fidelity to the full intervention, not merely which individual removal gives the smallest target norm. Any subsequent head selection must be labeled adaptive and checked on held-out examples/wording.

Audits: 128 native prompts, 800 interventions; exact identity replay; full native write reconstruction <=1.90e-7 relative; full four-condition interaction replay <=7.03e-6; component sum error <=1.87e-16; persistent-value interaction exactly zero; runner/manifest unchanged; finite metrics and all copying top1 correct. FineWeb remains 26,112 documents; broad goal active and incomplete.
