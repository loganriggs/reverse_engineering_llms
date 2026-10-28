## H583 — A strong agreement writer effect transfers across lexical items

Use precisely the five formats selected on H582 discovery before its transfer scoring. Reuse their prompts; this is a lexical-transfer retest at fixed wording, not new template transfer. Keep fixed candidate operations MLP3, MLP6 and Attention5 plus MLP2, MLP5, Attention4 and Attention6 controls. At final and penultimate positions, test bidirectional opposite-answer donor swaps, same-answer nuisance swaps and self controls; separately replay only the local-value effect through L4H0 or L7H8 as in H581.

312 token-valid prompts, 78 lexical pairs and 6,552 paired evaluations (13,104 intervention-stage forward passes). The Dennis/Jacqueline pair remains excluded in binding and copying by single-token answer eligibility. Three selected tasks pass both discovery and lexical-transfer competence gates: agreement, recall and copying. Binding and sentiment transfer cells remain failed and reported; they do not establish competent transfer.

The template field in result JSON encodes lexical split: 0 discovery, 1 transfer. It is not a second wording template in this experiment. All summaries retain site, operation and mode rather than selecting only the most favorable cases.

| Competent task | Operation | Discovery whole donor movement | Transfer whole donor movement | Transfer isolated-value movement | Transfer whole nuisance absolute movement |
|---|---|---:|---:|---:|---:|
|agreement|mlp_3|0.0544|0.0525|0.0007|0.0585|
|agreement|mlp_6|0.2160|0.2190|0.0340|0.0442|
|agreement|attn_5|1.2642|1.1495|0.0559|0.0408|
|agreement|attn_4|0.0489|0.0497|0.0054|0.0406|
|agreement|attn_6|0.0316|0.0196|0.0033|0.0129|
|capital_recall|mlp_3|0.0318|0.0314|0.0000|0.0622|
|capital_recall|mlp_6|0.0315|0.0326|0.0004|0.1466|
|capital_recall|attn_5|0.0206|0.0109|0.0001|0.2286|
|capital_recall|attn_4|0.0217|0.0216|0.0004|0.0669|
|capital_recall|attn_6|0.0711|0.0264|-0.0005|0.0678|
|sequence_copy|mlp_3|0.0069|0.0026|0.0005|0.0595|
|sequence_copy|mlp_6|-0.0016|0.0019|0.0002|0.0724|
|sequence_copy|attn_5|0.0148|0.0217|0.0028|0.0544|
|sequence_copy|attn_4|0.0297|0.0239|0.0012|0.0465|
|sequence_copy|attn_6|0.0408|0.0425|0.0008|0.0275|

The clearest result is Attention5 in agreement. Averaged over the two sites, donor-directed margin change is 1.264 discovery and 1.150 transfer, versus transfer nuisance absolute change 0.041. At the final “usually” token alone, mean donor-directed change is 2.369 discovery and 2.158 transfer; all 16 endpoint interventions in each split move in the predicted donor direction. At the penultimate token, means are 0.159 and 0.141. These are paired endpoints from only eight lexical items per split, not 32 independent population observations.

The final token is identical across singular/plural variants, so the effect cannot be attributed to exchanging that token's lexical identity. The earlier subject still differs in number and surface form, and nuisance swaps differ in content and magnitude; their smaller effects do not by themselves establish a pure number feature. Attention5 is the entire attention output, not one identified head or one feature.

The isolated L7H8 value path remains small: averaging both sites gives transfer donor movement 0.056 versus whole-write 1.150. This is not an additive mediation percentage. The principal agreement effect must be explained through additional routes or interactions under these interventions.

Recall and copying show much smaller donor-directed effects, often no larger than nuisance effects, at these fixed sites. Nearby operations also vary. Thus this supports a particularly useful agreement response at a specific operation/site, not a task-universal feature. Failed binding and sentiment transfer cells show donor movement but remain below their competence gates; do not promote them to confirmed transfer evidence.

Next use the replicated agreement effect to test a representation directly. Learn low-rank directions from discovery singular/plural Attention5 write differences at “usually”, freeze ranks/basis, then replace only the projected difference on every transfer item. Compare internal residual error against preservation of the full donor-swap output distribution (KL), grammatical-number margin and answer-group loss. Include norm-matched random subspaces and nuisance directions; retain all ranks and cases. If a compact shared direction preserves causal effects across nouns, test new wording and the other competent tasks before labeling it a reusable number feature. If internal and behavioral rank requirements differ, that directly informs the proposed data-based equivalence classes. This does not yet validate response-based grouping or a canonical basis.

Audits: native replay against H582 <=1.91e-5 margin; full self <=1.20e-5 and isolated self <=1.15e-5; finite metrics, source/manifest unchanged, exact locked-selection match. Routing code is inherited unchanged from H581's independent 40-case live-hook audit; no new independent routing audit is claimed for this panel. No new natural-text documents; FineWeb total 26,112. Broad goal remains active and incomplete.
