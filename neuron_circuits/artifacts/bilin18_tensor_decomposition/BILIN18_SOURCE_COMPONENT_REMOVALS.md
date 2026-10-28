## H973 — Causal source-position edits identify adjective MLP3 composition and period MLP4 transport

Previous turn was progress: H972 found substantial local MLP interaction at earlier adjective positions and ruled out factorial interaction in the layer0 value bypass. User explicitly requested continuing. This experiment independently removes local, inherited or total MLP interaction at adjective-last-subtoken and period positions for layers0–4. Each native prompt is replayed with a single position-specific MLP output replacement; all later attention, positions and layers recompute. Identity and paired norm-matched random factorial edits accompany every site/component.

All96 locked states retained. The following numbers concern the four familiar original-format quartets, the sole native-competent cohort. No per-example filtering or featurizer training. Source edits use teacher quartet components and may be off-manifold.

| Site/component removed | Intended-margin loss | Matched random RMS | Retained attention5 final-write interaction projection |
|---|---|---|---|
| Adjective MLP1 local |0.0602|0.0161|86.5%|
| Adjective MLP2 local |0.0571|0.0168|86.7%|
| Adjective MLP3 local |0.2640|0.0201|60.6%|
| Adjective MLP3 inherited |0.0828|0.0165|82.2%|
| Adjective MLP3 total |0.4366|0.0318|38.6%|
| Period MLP4 inherited |0.3517|0.0349|65.7%|
| Period MLP4 local |0.0038|0.0015|99.0%|

Attention5 retention is the new quartet interaction's signed projection onto its native interaction, averaged across quartets. It is not full-vector reconstruction fidelity or a causal mediation fraction, and can be outside[0,1]. The assay also tracks mixed-value interaction at adjective and period source positions.

Removing adjective MLP3 local interaction leaves60.4% of adjective-source value interaction and60.8% of period-source interaction at attention5, alongside60.6% of its final write. Total interaction removal leaves32.6%,39.9%,38.6% respectively. This makes adjective MLP3 a substantially stronger composition candidate than an algebraic norm-only screen.

Removing period MLP4 inherited interaction leaves21.6% of period-source value interaction and65.7% of attention5 final-write interaction; adjective-source interaction remains unchanged. This supports a downstream transport/transformation role at the period. Removing adjective MLP4 cannot change the same-block period state through tokenwise MLP computation; correspondingly the period source is unchanged at attention5 input under that edit. These causal-order patterns are consistent with the implementation, though they do not themselves establish semantic mediation.

Layer0 source-position interactions remain weak behaviorally despite large local algebraic shares: adjective local removal gives loss-0.0013 versus random RMS0.0157 and retains98.1% of attention5 interaction; period local removal is essentially inert. Thus large early cross-product norm alone was insufficient evidence. Period MLP0–3 local edits also have little effect. Candidate composition is more specifically concentrated around adjective MLP3, with smaller contributions at MLP1/2.

Candidate path, still incomplete: adjective-position local composition around MLP3; later transfer/transformation at the period, especially MLP4 inherited output; attention5 gathers composed values at the final token; later MLPs including MLP12 express/amplify the result. The source-to-period edge and path mediation require rescue or controlled edge interventions. Component effects are nonlinear and must not be added as percentages of total computation.

Audits:9,600/9,600 main forward examples plus16 independent manually assembled model forwards, total9,616. Manual source-position patches at layers0,2,3,4 and both roles/components replay saved margins within1.57e-5. Identity logit max difference2.46e-5; random norm mismatch2.17e-7 relative; identity downstream projection error5.40e-7; summaries independently recomputed; inputs unchanged and outputs finite. No training or reserved-test access; natural-document pool26,112.

Limits: small familiar competent cohort, privileged factorial edits, semantic/position confounds in the not contrast, and failed alternate-format cohorts. We now have evidence for a causally relevant local composition term, but not yet a reusable negation operator or a broadly validated manifold.

Next: rescue tests after adjective MLP3 local/total removal, restoring native downstream period MLP4 output or attention5 source contributions, with matched controls and native-rescue identity checks. Determine how much of the behavioral change is mediated through the proposed period/attention route before pursuing a learned reusable operator. Then test the frozen candidate on genuinely new competent task contexts. H961 frozen context-bridge predictor evaluation remains pending.
