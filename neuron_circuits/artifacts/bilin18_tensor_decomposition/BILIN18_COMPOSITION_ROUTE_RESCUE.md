## H974 — Downstream rescue supports a partial adjective-to-period-to-attention route

Previous goal turn was progress: H973 source-position removals identified adjective MLP3 local interaction and period MLP4 inherited interaction as causally relevant. User requested a high-level research update and continuation; the update distinguished useful result directions from actual composition sites and acknowledged the narrow competent cohort. This turn performs downstream rescue after adjective MLP3 local or total interaction removal.

Rescue options: restore the native period-position MLP4 output; restore native adjective, period, or both source contributions to attention5's final-token output; restore its full final-token output. Source correction is native source contribution minus currently perturbed source contribution, preserving all other sources and the attention bus. Every correction has +/- norm-matched random controls. Run all options without upstream ablation as native-rescue identity checks. Rescues restore whole outputs/contributions, not just their factorial coefficients.

All96 locked states retained. Numbers below concern four familiar original-format quartets, the sole native-competent cohort. Margin recovery=(ablation loss-rescued loss)/ablation loss. KL remaining is KL(native||rescued) divided by KL(native||ablated), aggregated over endpoints. These are conditional rescue results, not unique or additive mediation percentages.

| Native component restored | Margin recovered after local ablation | KL remaining after local ablation | Margin recovered after total ablation | KL remaining after total ablation |
|---|---|---|---|---|
| Period MLP4 write |56.1%|0.262|59.2%|0.229|
| Attention5 adjective source |20.6%|0.568|18.9%|0.596|
| Attention5 period source |27.0%|0.579|27.6%|0.559|
| Attention5 both sources |45.4%|0.293|43.8%|0.304|
| Attention5 full final-token write |45.8%|0.297|44.3%|0.306|

Unrescued margin losses are0.2640 for local ablation and0.4366 for total. Period MLP4 restoration reduces these to0.1158 and0.1783. Its random corrections recover about-1% for local ablation and-1% to-2% for total. Attention-source random controls also remain nearzero recovery (about-2.3% to+2.4%). Thus recovery is specific to the native correction, beyond equal-sized random edits in this limited control set.

Restoring both adjective and period source contributions performs almost as well as restoring the complete attention5 final-token write, supporting their importance for this downstream branch. Restoring the period MLP4 write helps more than restoring attention5's final write, plausibly because the period influences additional downstream paths. That explanation is an inference, not directly separated mediation. Neither rescue completely restores behavior, so the candidate route is partial. Source-rescue effects are nonlinear and must not be summed as path fractions.

Audits:4,608 main plus16 independent manually assembled forwards, total4,624. Native-rescue identity maximum logit error2.67e-5, including random corrections when true rescue is nearzero. Dynamic attention-source contributions reconstruct actual outputs within2.92e-7 relative; random norm mismatch1.54e-7. No-rescue ablations independently replay H973 margins within6.56e-6. Independent manual period-write/full-attention rescue at four competent positive/plain endpoints and both upstream ablations agrees within6.70e-6; summary aggregation checked. Inputs unchanged, outputs finite; no training or reserved-test access; natural-document pool26,112.

High-level interpretation: causal evidence now links a local bilinear composition term at adjective MLP3 to a later period representation and attention5 transport, followed by late output expression. This is more specific than a shared feature direction and supplies a candidate computation for future featurizer evaluation. It is not yet a generally reusable negation operator or a validated unsupervised manifold: four familiar quartets, privileged teacher edits, and failed alternate formats remain substantial limitations.

Next priority: freeze the candidate sites and formulas, then create a prospective factorial context/lexicon panel and screen native competence before applying the locked interventions/rescues. Do not retune sites or discard failures to create apparent generalization. Distinguish exact new adjective pairs, new noun contexts, and altered output wording. Return to multi-task featurizer comparisons using a causally validated target only if this transfer is supported. H961 frozen context-bridge predictor evaluation remains pending.
