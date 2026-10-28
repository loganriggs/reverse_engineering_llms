## H589 — Frame-dependent attenuation mostly arises before the terminal readout; compensation can conceal it

Capture the native final residual and the response to identical number-axis perturbations at +/-0.0625 and +/-0.5 discovery units for all 192 prompts. This uses 960 model forwards including native captures. For each matched lexical/type/subject-number/attractor-number combination, cross three donor-frame final response vectors with three recipient-frame native final residuals. Apply the model's unchanged final RMS normalization, unembedding, tanh cap and answer-group measurement. There are 2,304 terminal residual crossings; the 768 same-frame diagonals must replay H588.

An off-diagonal state is recipient_native_final_residual + (donor_perturbed_final_residual - donor_native_final_residual). This isolates a propagated response from its terminal readout baseline; it is a counterfactual state, not a naturally generated prompt. Responses from different frames are not asserted semantically interchangeable outside this test.

At small dose, rows below specify recipient terminal context and columns specify the frame that produced the response. Values are mean central plural response per discovery unit.

| Type / terminal context | Response from preposition | Response from relative clause | Response from fronted phrase |
|---|---:|---:|---:|
|regular/preposition|1.2763|0.9881|0.6835|
|regular/relative_clause|1.5235|1.2250|0.8520|
|regular/fronted_preposition|1.2294|0.9341|0.6522|
|irregular/preposition|1.2786|0.8754|0.6714|
|irregular/relative_clause|1.4852|1.0620|0.7893|
|irregular/fronted_preposition|1.2628|0.8726|0.6711|

For regular nouns, the native fronted response is 0.652 versus unfronted 1.276. Putting the unfronted response into the fronted terminal context gives 1.229; putting the fronted response into the unfronted terminal context gives only 0.683. Irregular nouns show the same pattern (1.263 and 0.671 respectively). Thus most of this frame contrast follows the propagated response vector, rather than terminal-readout sensitivity alone. The half-dose crossings agree qualitatively.

Relative clauses reveal compensation. For regular nouns, native responses look similar: 1.276 unfronted versus 1.225 relative clause. But a relative-clause response read in the unfronted terminal context gives only 0.988, while the unfronted response read in the relative-clause terminal context gives 1.523. Intermediate attenuation and stronger terminal sensitivity offset. Similar observable effects therefore need not imply the same internal computation.

An exact symmetric allocation of each 2x2 cross is saved: propagation contribution = half of the donor-response change under each terminal context; terminal-context contribution = half of the context change under each donor response. These sum to the diagonal contrast by construction; interaction is reported separately. At small dose, regular fronting has total change -0.624, allocated -0.585 propagation and -0.039 terminal context; irregular fronting -0.607 = -0.599 + -0.008. For relative clauses the opposing terms are -0.293 +0.242 (regular) and -0.413 +0.197 (irregular). These are algebraic summaries of this crossing, not independent causal percentages or unique mediation shares.

Research implication: a common residual direction has context-dependent downstream transformations. A data-based equivalence class defined only by final margin can merge states with different internal responses and compensation. That may be acceptable for a behavioral objective, but cannot establish mechanistic equivalence. We still have not identified the later operations that attenuate, rotate or compensate for the number response.

Next trace the perturbation's residual response through later blocks, accounting exactly for residual scaling and attention/MLP write changes. Use that accounting to select targeted response clamps on discovery items and test on irregular/held-out items. Prioritize explaining the fronted attenuation and relative-clause compensation rather than adding more descriptive compression. Keep fixed-input and context-crossing predictions distinct.

Audits: 768 diagonal crossings replay H588 within 1.63e-5 margin; symmetric decomposition closes exactly; finite metrics, unchanged source/basis. No new prompts or natural-text documents; FineWeb total 26,112. Broad goal active and incomplete.
