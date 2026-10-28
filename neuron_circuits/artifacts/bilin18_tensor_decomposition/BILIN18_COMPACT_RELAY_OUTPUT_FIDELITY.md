## H950 — Concentrated products preserve downstream behavior on the anchored task panel

Test: evaluate every H949 penalty/seed, both complete 64-product fits and discovery-selected top-eight subsets, in the live downstream model. All 312 existing full/rank-one input interventions retained. No parameters or subset coefficients refit. Subsets are task-specific and selected using discovery responses only. The native recipient MLP update stays intact; the predicted finite MLP change is added either alone or with the exact raw residual change. Thus this tests causal-effect approximation, not replacement of the native MLP.

All 11,232 whole-model forwards completed. Table entries span the two seeds. Task effect error is RMS margin-change error divided by reference RMS. KL ratio is full-output KL versus exact intervention, divided by the KL of omitting the MLP change on the same path; smaller is better. A zero MLP change has KL ratio one.

| Input intervention | Task | Penalty | Products | Joint effect error | Joint KL ratio |
|---|---|---|---|---|---|
| full | agreement | 0.0 | all64 | 23.459–23.565% | 0.113–0.115 |
| full | agreement | 0.0 | top8 | 58.440–64.337% | 0.634–0.694 |
| full | agreement | 0.01 | all64 | 25.334–25.575% | 0.131–0.131 |
| full | agreement | 0.01 | top8 | 31.454–39.805% | 0.198–0.301 |
| full | agreement | 0.1 | all64 | 24.402–25.911% | 0.122–0.136 |
| full | agreement | 0.1 | top8 | 23.283–25.509% | 0.113–0.133 |
| full | agreement | 1.0 | all64 | 79.730–79.771% | 0.999–1.001 |
| full | agreement | 1.0 | top8 | 79.722–79.841% | 0.999–1.001 |
| full | sentiment | 0.0 | all64 | 27.553–27.578% | 0.100–0.102 |
| full | sentiment | 0.0 | top8 | 49.830–64.532% | 0.302–0.524 |
| full | sentiment | 0.01 | all64 | 28.491–28.521% | 0.107–0.109 |
| full | sentiment | 0.01 | top8 | 32.152–32.659% | 0.135–0.140 |
| full | sentiment | 0.1 | all64 | 28.164–28.343% | 0.104–0.107 |
| full | sentiment | 0.1 | top8 | 26.762–27.466% | 0.097–0.102 |
| full | sentiment | 1.0 | all64 | 90.246–90.302% | 1.000–1.001 |
| full | sentiment | 1.0 | top8 | 90.246–90.271% | 1.000–1.000 |
| rank1 | agreement | 0.0 | all64 | 11.489–11.985% | 0.039–0.043 |
| rank1 | agreement | 0.0 | top8 | 55.096–61.425% | 0.611–0.669 |
| rank1 | agreement | 0.01 | all64 | 13.815–14.776% | 0.053–0.060 |
| rank1 | agreement | 0.01 | top8 | 23.592–31.451% | 0.131–0.219 |
| rank1 | agreement | 0.1 | all64 | 13.886–15.629% | 0.056–0.067 |
| rank1 | agreement | 0.1 | top8 | 13.620–16.243% | 0.055–0.071 |
| rank1 | agreement | 1.0 | all64 | 77.512–77.517% | 1.000–1.000 |
| rank1 | agreement | 1.0 | top8 | 77.493–77.497% | 1.000–1.000 |
| rank1 | sentiment | 0.0 | all64 | 17.132–17.387% | 0.047–0.048 |
| rank1 | sentiment | 0.0 | top8 | 42.550–58.078% | 0.235–0.447 |
| rank1 | sentiment | 0.01 | all64 | 17.762–17.991% | 0.050–0.051 |
| rank1 | sentiment | 0.01 | top8 | 21.640–22.264% | 0.071–0.074 |
| rank1 | sentiment | 0.1 | all64 | 18.402–18.911% | 0.053–0.056 |
| rank1 | sentiment | 0.1 | top8 | 18.963–19.000% | 0.057–0.057 |
| rank1 | sentiment | 1.0 | all64 | 87.987–88.025% | 0.999–1.000 |
| rank1 | sentiment | 1.0 | top8 | 88.005–88.034% | 0.999–1.000 |

Main result: for the .1 penalty, top eight products yield full-intervention joint effect errors of 23.3–25.5% agreement and 26.8–27.5% sentiment. Joint KL ratios are .113–.133 and .097–.102 respectively, similar to the complete fit. The top-eight subsets are sometimes slightly better than the full fit on this panel; this is descriptive, not a separately confirmed benefit of pruning. Without the contribution penalty, top-eight errors are 58–64% agreement and 50–65% sentiment, much worse than full fits. The .01 penalty is intermediate. The strongest penalty again behaves like omitting the MLP change.

The exact residual bypass helps joint responses, so isolated MLP effects matter too: .1/top-eight isolated errors are 44–49% agreement and 33–34% sentiment, with KL ratios .306–.330 and .118–.123. Thus substantial teacher mismatch remains. Compactness is compatible with preserving the learned approximation; it has not made that approximation complete.

Audit: checkpoint finite effects replay exactly; whole-product sum versus direct floating-point model has relative error 2.30e-6. Native identity logits match historical captures to 3.43e-5 maximum absolute error; exact joint intervention margins match historical results to 1.79e-5. All expected forwards present, finite records, hashes unchanged, independently reaggregated metrics agree within 1e-12. Reserved test untouched; unique natural documents remain 26,112.

Interpretation: supervised causal anchors plus a contribution penalty can produce a compact approximation of these two mechanisms, with downstream distribution fidelity maintained. This is not yet an unsupervised feature ontology, general MLP compression, unique units, or proof of cross-task reuse. Evaluation uses the existing development panel, so a new transfer panel remains necessary before strong generalization claims.

Next: determine whether concentration makes the dominant components correspond across independently trained seeds, using tensor and discovery-functional matching with held-out response checks. Then freeze candidate components and test novel contexts/combinations, including failures and off-target effects. Avoid further tuning on this development panel.
