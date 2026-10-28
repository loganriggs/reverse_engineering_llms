## H588 — Identical number-axis perturbations have context-dependent downstream effects

Previous goal turn made progress by crossing number, morphology and noun order, but left write amplitude versus downstream sensitivity confounded. Apply exactly the same signed residual perturbations at Attention5's final write across all 192 H586/H587 prompts. Frozen PC1 is oriented toward the original discovery plural-minus-singular mean. Its discovery-scale amplitude is 669.075, fixed before this test. Doses are +/-0.0625, +/-0.25, +/-0.5 and +/-1.0 times that amplitude. Nuisance and five seeded random unit directions receive identical norms; zero is evaluated once per prompt. Total 10,944 interventions, no prompt-specific rescaling or refitting.

Central plural response is [plural_shift(+dose)-plural_shift(-dose)]/(2*dose), where plural_shift is the decrease in the singular-minus-plural answer-group margin. It is expressed per original discovery-scale unit, not per raw residual unit. This is a finite-difference response, not an exact derivative. All existing competent cells and individual mistakes remain.

| Type/frame | Small-dose central response | Positive fraction | Half-dose response | Full-dose response |
|---|---:|---:|---:|---:|
|regular/preposition|1.2763|1.0000|1.3610|1.4168|
|regular/relative_clause|1.2251|0.9688|1.2914|1.3028|
|regular/fronted_preposition|0.6522|0.8438|0.7493|0.8906|
|irregular/preposition|1.2786|1.0000|1.3489|1.3777|
|irregular/relative_clause|1.0620|0.8750|1.1640|1.2272|
|irregular/fronted_preposition|0.6711|0.8125|0.7432|0.8431|

Fronted-frame sensitivity is roughly half the unfronted value even with identical input perturbations. H586/H587 also showed smaller written number coefficients there. Therefore both encoding amplitude and downstream sensitivity contribute to the smaller donor-swap effect. This is not an additive percentage decomposition.

The sign is not universal at small dose: 16 of 192 cases have negative central responses, all with singular subjects. Responses range -0.876 to +3.440 per discovery unit, so the reversals are not merely the replay tolerance. At full dose the central response is positive everywhere, but even-response terms grow substantially: dose dependence and asymmetry matter. Do not model the axis as a context-invariant linear plural knob.

Number-axis mean small-dose responses exceed the absolute means of the equal-norm nuisance/random controls in every type/frame cell; controls are not exactly inert. Signed random-axis orientation is arbitrary, so compare magnitudes. Native-to-intervened KL and intended-answer-group NLL changes are retained separately; larger linguistic margin effects are not a complete description of distributional change.

Next H589 crosses the actual final residual response vectors with each frame's native final residual. This will distinguish changes accumulated by later blocks from changes due to the terminal normalization/unembedding/tanh/answer-group readout context. It is a diagnostic intervention, not proof of one uniquely responsible gate.

Audits: zero <=1.34e-5 margin; all directions unit-normalized within 1.79e-7; finite metrics and frozen source/basis hashes. No new prompts or natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
