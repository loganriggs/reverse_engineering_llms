## H576 — A single producer head retains the punctuation-dependent transfer failure

Freeze H575's selected heads: binding L9H7/L9H6, copying L7H8/L4H0. Test on H572's 128 new-template edited cases. Conditions are each of the four individual heads, own-task pair, other-task pair, all-four union, whole parent attention layer(s), MLP10 and self: ten modes, 1,280 interventions. The prompt panel was already characterized; these are new interventions, not independent new lexical examples. No head choices are updated after seeing transfer results.

| Condition | Own head pair | Other head pair | Four-head union | Whole parent layer(s) |
|---|---:|---:|---:|---:|
|binding_new_paren|0.824|0.941|0.752|0.794|
|binding_new_comma|0.863|0.937|0.796|0.790|
|copying_new_paren|0.475|1.242|0.652|0.475|
|copying_new_comma|1.295|0.909|1.078|1.138|

Binding's selected pair removes about 14–18% of the gate correction; this is partial transfer. Copying's pair removes about 52% with parentheses but increases comma discrepancy about 29%. Whole-layer mixing is not the sole cause: L4H0 alone reduces the new copying parenthesis discrepancy about 33%, but increases comma discrepancy about 26%. L7H8 helps parentheses about 20% and has little comma effect.

L9H6 supplies a second counterexample to fixed head roles: its clamp reduces binding correction modestly, while increasing new-template copying-parenthesis discrepancy about 26%. The four-head union can therefore be less effective than the copying pair alone in that condition. More components do not automatically give a better explanatory circuit.

These effects do not establish whether the head's native contribution is helpful or harmful for task accuracy, nor whether its gate effect mediates its whole output effect. They show that the operation attributed to an isolated producer head depends on context and downstream consumers. A head name or a task label is not yet a stable semantic/computational unit.

Next H577: decompose the selected copying producers, especially L4H0, by source-token groups (inserted span, selected name, earlier names, other prefix, and suffix/self positions), including both local and persistent value contributions. Verify that their sum reconstructs the head write before applying source-group ablations. Compare both templates and punctuation forms. This tests whether the change in effect comes from what the head reads or from how later computation interprets its write. Retain L7H8 as a control and distinguish source-group evidence from a full circuit explanation.

Audits: head sums reconstruct within 2.52e−7 relative error; raw-query reconstruction <=1.98e−6; self margin <=7.63e−6 and coefficient <=9.98e−7; finite outputs and unchanged source. H575–H576 complete 5,376 interventions on reused prompts, no new FineWeb documents; tracked total 26,112. Broad goal active and incomplete.
