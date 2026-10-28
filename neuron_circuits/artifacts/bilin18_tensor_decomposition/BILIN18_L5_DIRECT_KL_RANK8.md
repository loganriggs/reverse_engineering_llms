## H773 — Direct-KL training works numerically, but the rank-eight pilot does not improve pooled validation fidelity

Previous goal turn was progress: H772 evaluated frozen sensitivity-weighted fits on additional documents and found no general gain. H773 tests the more direct objective with a matched, simple representation before adding sparse gates.

The representation is a rank-eight linear encoder/decoder in fixed output-write coordinates, with a fixed natural-data mean and scale. Encoder and decoder together contain2,048 trainable parameters; rank stays at most eight. Initialization is the original large-natural-data write-PCA rank-eight reconstruction. Both treatments use exactly that initialization,128Adam steps, learning rate.001, batches of8, and the same fixed minibatch schedule. Each objective is divided by its own first training-batch initial loss, fixed thereafter. The base language model is frozen.

Training uses128 predetermined natural training documents at prefix lengths32 and120 (256contexts). One treatment minimizes write-space reconstruction MSE; the other backpropagates exact KL(native output distribution||edited output distribution) through the downstream model. Teacher probabilities include the full vocabulary. Both train encoder and decoder. There is one schedule and no validation-selected checkpoint or hyperparameter search. Final checkpoints alone are evaluated, alongside untouched initialization, on the same64document/128context research-validation panel as H772.

| Objective | Training mean KL | Validation mean KL | Validation scaled write MSE |
|---|---:|---:|---:|
| initial | 0.00248105 | 0.00261506 | 0.959455 |
| reconstruction | 0.00220301 | 0.0028081 | 0.975755 |
| direct_KL | 0.00132217 | 0.00271341 | 0.988026 |

Direct KL training lowers training KL by46.7%, but pooled validation mean KL is3.8% above initialization. Reconstruction training lowers training write error but raises validation KL by7.4%. Direct-KL validation KL is3.4% below reconstruction training, but all three paired comparison intervals include zero under a descriptive10,000-draw document bootstrap. No reliable generalization gain has been established.

Length-specific results differ: direct-KL mean validation KL improves from.001045 to.000891 at length32, but worsens from.004185 to.004536 at length120. The pooled result should not be described as a universal degradation; it is also not a reason to choose a favorable length after seeing outcomes. Direct KL is lower than initialization on56.25% of validation documents, while the mean is worse, indicating heterogeneous magnitudes. Full paired comparisons and length summaries are saved.

Observed-token CE change is measured separately. Validation mean CE changes relative to the native model are+.00241nats at initialization,+.00381 for reconstruction training, and+.00462 for direct-KL training. No CE improvement claim follows. Scaled write MSE is the mean squared error in globally scaled write coordinates, not a normalized variance-explained score.

Implementation checks precede training. The end-to-end encoder KL gradient agrees with central finite differences along its normalized gradient direction to0.032% and0.094% at two step sizes on the audited batch. Identity replacement gives zero KL and gradient maximum5.1e-19. Full-rank reconstruction replays logits within1.42e-5. Native head capture agrees with the earlier cache within5.35e-5. All model parameters remain frozen; all output metrics are finite; saved final-fit and source/input hashes stay unchanged. The numerical gradient audit is local to the tested batch/direction, not an exhaustive proof of every gradient.

Compute:2,616 whole-model forwards and1,040 context-level KL backward evaluations. Training/validation documents are disjoint. No original reserved-test evaluation; unique natural document pool remains26,112. The KL normalizer differs from the reconstruction normalizer, so equal steps and learning rate are a controlled pilot, not proof of equal convergence or optimal tuning for each objective. One initialization/schedule and128steps leave optimization and sampling uncertainty unresolved.

Interpretation: the direct loss and gradient path are functional, but there is no verified better representation to interpret as a manifold. A substantial training improvement paired with no pooled validation improvement makes generalization a more immediate issue than adding hierarchy or DAG structure. The initialization was learned from much more natural data than these finetuning objectives; that asymmetry matters even though the two finetuning treatments are matched.

Next: test training coverage explicitly before claiming the objective fails. Use nested128- and1,024-document training sets with the same rank, initialization, learning rate and128-step exposure budget; retain both reconstruction and KL objectives and repeat fixed minibatch seeds. Freeze the comparison before evaluation, report all conditions and lengths, and use no validation checkpoint selection. Increasing document variety at fixed update count distinguishes exposure diversity from simply training longer. If broader coverage still fails to improve actual KL, do not advance the fit into causal/manifold claims merely because training loss falls.
