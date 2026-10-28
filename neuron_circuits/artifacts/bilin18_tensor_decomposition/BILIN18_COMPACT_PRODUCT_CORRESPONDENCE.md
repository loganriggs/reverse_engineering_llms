## H951 — Causal contribution concentration improves dominant-product reproducibility

Question: does H949's penalty merely compact responses arbitrarily, or do independently trained models acquire corresponding products? All four penalties and both seeds retained. No refitting. Two Hungarian matchings use either whole symmetric bilinear tensors or full product response matrices on discovery anchors (equal task energy normalization), then evaluate on existing held-out full and rank-one contrasts. Independently selected dominant products are also compared directly, without matching on evaluation data. A whole-product response includes both its scalar response and output vector.

For the independently discovery-selected dominant product, held-out full-intervention response cosine across seeds is:

| Penalty | Agreement | Sentiment |
|---|---|---|
| 0 | .004 | .137 |
| .01 | .560 | .683 |
| .1 | .894 | .982 |
| 1 | .003 | .107 |

At .1, dominant response amplitudes still differ: seed1/seed0 norm ratios .847 agreement and .941 sentiment; relative response disagreement is 45% and 19%, respectively. Teacher errors remain 74–79% agreement and 58–60% sentiment for these single products. Reproducibility is not teacher fidelity.

The .1 dominant sentiment pair has whole tensor cosine .585, versus agreement .095. Their behavior is more similar on the observed data than their global bilinear functions. Factoring response similarity confirms scalar finite-response cosine .990 agreement/.9997 sentiment and output-direction cosine .903/.982 (absolute values remove compensated sign gauge). This suggests residual disagreement lies substantially in output direction and amplitude, while the observed scalar response profiles align. These task-conditioned, correlated contrasts do not identify behavior on new input directions.

Eight-product sums at .1 are even more reproducible: cosine .987 agreement/.997 sentiment and relative disagreement 16%/8%. Functional matching gives energy-weighted individual product cosines .861/.979, but unweighted means only .055/.241. Therefore the result is carried by dominant products; the small residual products remain unstable. Giving all products equal weight would obscure this distinction. The weighted statistic uses evaluation energy for descriptive reporting only, never for selecting the matching or subset.

Reuse check: the dominant product index differs between agreement and sentiment in both seeds. Agreement-dominant product response norms are 48–56% of the agreement teacher norm but only 1–10% of the sentiment teacher norm. Sentiment-dominant products are 69–73% on sentiment and 4–12% on agreement. These are response-norm ratios, not explained-variance fractions. The top-eight sets overlap in 5–6 indices, but this includes weak residual products and does not establish shared computation. The evidence favors task-selective dominant responses on this panel.

Audit: functional self-cosines within 1.23e-15 of one; explicit materialized response-vector cosines agree with factored Gram formula within 6.67e-16. All assignments are permutations; inputs unchanged; finite outputs. Product response cosine independently factors into scalar and output cosines within 1e-12. Zero new model forwards; no reserved-test use; natural document pool unchanged at 26,112. Only two seeds, supervised anchors, and an extensively reused development panel limit the conclusion.

Conclusion: adding a causal contribution prior improves not only compactness but also dominant functional reproducibility in this setting. It has not established unique global units, a broad ontology, or cross-task computational reuse. This is an informative advance beyond arbitrary coordinate stabilization: held-out intervention responses, not merely latent labels, align more strongly. However both models learned the same narrow anchor tasks, so a fresh transfer test is essential.

Next: freeze .1 as an exploratory candidate together with unpenalized and .01 controls, preserve all other fits as diagnostic controls, and evaluate novel competent contexts/combinations with native teacher effects. Do not further optimize penalties on the existing panel. Include task cross-interventions and output distribution checks; prioritize new input directions over more paraphrases of the same templates.
