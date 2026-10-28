## H738 — Agreement uses contextual local values; recall-head effects largely follow the persistent token-value bus

We decomposed the native computation at fixed L11H3 and L13H0, rather than fitting another compression model. For the final query, each head computes H=A V, where its signed routing pattern is A=(QK/128)*(Q2K2/128), after the model's RMS normalization and rotary transforms. Values are V=(1-lambda)L+lambda B: L is the current layer's contextual value vector, and B is the persistent layer-0 value bus. Neither A nor the mixing coefficients are probabilities.

The learned mixing parameters are lambda=-1.75 at L11H3 and lambda=4.1875 at L13H0. Thus local and bus terms have opposite signs: 2.75 L-1.75 B for agreement, and -3.1875 L+4.1875 B for the recall head. Large mixing coefficients alone do not establish causal importance; the interventions below measure it.

Using recipient r and donor d, the exact head-vector change decomposes as:

- Routing-only: (A_d-A_r)V_r.
- Values-only: A_r(V_d-V_r).
- Bilinear interaction: (A_d-A_r)(V_d-V_r).

The values-only change further splits into current-layer and persistent-bus contributions. Each term is added separately to the native receiver head, with downstream computation live. These head-vector terms add exactly; their final output effects need not.

Eligibility was fixed before intervention outputs: recipient/donor sequences must have equal token length and are aligned by absolute position. Of 231 intended directions, 198 pass and 33 are excluded (six task directions and 27 nuisance directions). All task-direction exclusions are in the already-failing sentiment family; competent task cells retain their full H732 task-pair sets. Many nuisance prompts change sequence length, so nuisance availability differs and no matched-nuisance superiority claim is made from this factorial alone. Equal token length is a positional intervention convention, not semantic realignment.

Mean fraction of the exact-head donor-directed margin effect, competent strata:

| Head and task / format | Routing only | Values only | Local values only | Bus values only | Interaction only |
|---|---:|---:|---:|---:|---:|
| Agreement, agreement A | 0.9% | 98.1% | 104.1% | -1.6% | 1.5% |
| Agreement, agreement B | 0.3% | 98.4% | 102.4% | -1.6% | 4.3% |
| Recall, capital recall B | 0.1% | 100.0% | 3.4% | 95.3% | -0.1% |
| Recall, entity binding A | 11.6% | 89.3% | 2.5% | 86.9% | -1.5% |
| Recall, entity binding B | 9.0% | 90.6% | 2.6% | 88.1% | -0.5% |
| Recall, sequence copying A | 2.0% | 98.5% | 7.6% | 89.9% | 1.2% |
| Recall, sequence copying B | -0.9% | 100.3% | 6.1% | 93.4% | 1.1% |

Fractions are output-margin effects, not accuracy. Individual columns need not sum to 100% because they are separate nonlinear downstream interventions. Values above 100% and negative effects are retained rather than clipped. Capital Format A and sentiment remain archived diagnostic strata because native competence failed.

The agreement contrast is predominantly carried in contextual local values; the bus contribution slightly opposes it. The recall head is different: persistent-bus changes explain most of its measured effects across factual recall, binding and copying. Routing changes matter somewhat more for binding than for the other tested contrasts.

A small routing-only effect does not mean routing is unnecessary. The values-only experiment uses the receiver's existing context-dependent route to select and sign the values. It shows that substituting donor values while holding that route fixed often approximates the donor head effect. Both routing and local values may also contain upstream computation; these interventions localize contributions at the chosen head, not the origin of the whole task solution.

This supports investigating a shared read operation over a distributed token/value map, instead of labeling high-capacity sparse atoms as fundamental units. The layer-0 bus is generated before attention mixes positions, but the meaning of the head's output map still needs testing: identity copying, identity suppression, and lexical/factual association are different hypotheses. Prior reader results for L14H4/L8H3 motivated the method but are not being counted as evidence for L13H0.

Audits: native final logits replayed exactly; reconstructed head vectors matched native outputs within 1.10e-6 relative error. The full routing/value vector identity held to 3.61e-15 and the local/bus identity to 2.00e-16. Reconstructed full-head swap effects matched H732 within 9.54e-6; zero-edit margin error was at most 7.15e-6. Two independent algebra tests passed, including signed nonconvex mixing and fixed-factor limits. All input/source hashes remained unchanged and results were finite. The run used 3,003 full-model example forwards, no new natural documents and no original reserved-test examples.

Next: characterize L13H0's fixed bus-to-output map separately from its signed contextual coefficient. On structurally factorable source substitutions, test whether the fixed map predicts identity/association contrast and whether the native coefficient predicts the direction of the observed bus intervention, without fitting signs on these outcomes. Then use absent-token substitutions to distinguish lexical identity handling from task-answer-specific behavior. Keep routing-change and local-value residuals explicit, retain all competence labels, and carry forward matched controls. This is an operational test of reuse, not yet a manifold or semantic-feature claim. Sentiment's format/content diagnostic remains outstanding.
