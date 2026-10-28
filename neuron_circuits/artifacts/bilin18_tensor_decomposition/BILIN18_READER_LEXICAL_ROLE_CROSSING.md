## H541 — Reader sign changes survive matching names, final token and source positions

The previous goal turn made progress: H539/H540 supported conditional signed identity readers, while leaving a lexical-cue alternative. Binding ended in “to” and copying used different final key words. This experiment uses the same eight name pairs and the final token “to” in both tasks, retaining the four frozen heads and discovery identity-map signs.

Remeasure the 24 binding anchor pairs from H540 (48 prompts). Add five copying formats per name pair: “to A; to A; to”, a prefixed version, three repetitions, a version padded to binding's token length, and a version with “and back” after each repeated value. The last version is padded so its final token and nearest repeated-name position exactly match the binding final token and giver position. In this strongest condition, intervene only on that nearest repeated name; other copying conditions intervene on all changed value occurrences. Padding is the declared repeated word “Note”, selected by token count alone before inference. This is a position control, not naturalistic prose.

All eight formats pass the competence gate. Every copying format has 100% both-endpoint contrast correctness and 100% full-vocabulary top-1. Binding anchors have 100% both-endpoint correctness and 87.5–100% full-vocabulary top-1. No pair is excluded. There are 80 new copying prompts and 48 repeated binding prompts, not 128 wholly fresh prompts.

### Routing and causal identity effects

Use the same name-valued third-token pairs across both tasks. At native gate +1, alter only the designated source-to-head persistent-bus contribution and propagate through live downstream computation. Measure third0-versus-third1 output contrast under their respective injections. Positive effects mean relative identity copying; negative effects mean relative identity suppression. No gate reversal or zero-gate condition is rerun here.

| Context | Reader | Positive routing coefficients | Mean third-token identity effect | Locked conditional-sign predictions |
|---|---|---:|---:|---:|
|entity_binding/position_shift|L14H4|16/16|-3.3514|16/16|
|entity_binding/position_shift|L8H3|16/16|-0.2908|16/16|
|entity_binding/reversed_names|L14H4|16/16|-3.1085|16/16|
|entity_binding/reversed_names|L8H3|16/16|-0.3283|16/16|
|entity_binding/standard|L14H4|16/16|-3.0854|16/16|
|entity_binding/standard|L8H3|16/16|-0.2830|16/16|
|sequence_copy/length_matched|L14H4|0/16|2.4164|16/16|
|sequence_copy/length_matched|L8H3|0/16|0.9964|16/16|
|sequence_copy/position_matched|L14H4|5/16|0.0973|16/16|
|sequence_copy/position_matched|L8H3|0/16|0.3137|16/16|
|sequence_copy/prefix|L14H4|0/16|2.3697|16/16|
|sequence_copy/prefix|L8H3|0/16|1.2832|16/16|
|sequence_copy/standard|L14H4|2/16|1.3233|16/16|
|sequence_copy/standard|L8H3|0/16|1.3289|16/16|
|sequence_copy/three_repeats|L14H4|0/16|2.8999|16/16|
|sequence_copy/three_repeats|L8H3|0/16|2.4583|16/16|

L8H3 uses a positive coefficient and suppresses identity in every binding anchor, while using a negative coefficient and copying identity in every copying prompt. This persists in all 16 strongest matched-position directions: source name, final token, source position and final position are identical across the paired task contexts. A fixed response to the final token or absolute source/query positions alone cannot explain this sign change.

L14H4 is less uniform. In the strongest matched condition, 5/16 coefficients are positive and the average third-token effect is only +0.097, versus −3.085 in standard binding. It does not implement an unconditional “copy in copying tasks” rule. The coefficient-conditioned identity rule still predicts all leading-head effects correctly (256/256 across this run), including these exceptions. Perfect native copying in those contexts shows that the full task does not require this one reader to choose the copying sign.

Both leading readers keep their second query/key factor positive at the tested sources. The first factor supplies the context-sensitive sign change. Query and key vectors are contextual representations, even though the selected tokens and their positions match. That motivates separating their contributions rather than attributing the gate to a word label.

### Limits and next test

This rejects final-token identity and absolute position as complete explanations for L8H3 on the tested panel. It does not isolate an abstract relational variable: narrative versus repeated-list structure, surrounding words, padding, and earlier representations still differ. L14H4's exceptions further constrain a broad role interpretation. The result supports a context-derived gate, not a universal “giver” feature.

Next: use the saved native projections from the standard binding and position-matched copying contexts. Cross query origin (binding/copying) and key origin (binding/copying), replacing actual projections at the matched head/positions. Compare the two query/key factors and causal third-token effects. Keep both factor pairs and all heads frozen; do not fit a new semantic direction. This tests whether contextual information at the query, the source key, or their combination selects the sign.

Audits: all 16 matched lexical directions have exactly equal selected source token, final token, source position and final position across tasks. 1,664 interventions and 512 third-token comparisons; token-value replay error <=1.15e-5, coefficient factorization relative error <=2.1e-7, self margins agree within 5.3e-6. Outputs finite and source/manifest hashes unchanged. Native projections and all prompt/token records are saved for the factorial follow-up. No new FineWeb documents; tracked total 26,112.
