## H774 — Broader training coverage reduces the pilot's validation loss, but the direct-KL gain remains small

Previous goal turn was progress: H773 audited direct-KL gradients and exposed a training/validation gap. H774 varies training coverage while preserving the rank-eight parameterization, original PCA initialization, batch size8,128update budget, Adam learning rate.001, and loss normalizers. It includes both reconstruction and direct-KL objectives and three prespecified schedules (7731,7742,7743). No final model or checkpoint is selected using validation.

The128-document pool is nested inside the1,024-document pool. Both use lengths32 and120. At fixed1,024 context draws per training run, the small pool exposes127–128distinct documents and248–253distinct document/length contexts; the large pool exposes637–645documents and793–799contexts. This tests increased exposure diversity at constant update count, not complete coverage of all1,024documents. The natural PCA initialization and coordinate scale remain identical across runs.

Validation is the same64document/128context research panel used in H772/H773. The table shows means across the three schedules; each individual schedule is also retained in the data.

| Training pool | Objective | Mean validation KL | Range across schedules |
|---|---|---:|---:|
| 128 | reconstruction | 0.00281479 | 0.00280431–0.00283195 |
| 128 | direct_KL | 0.00266321 | 0.00263546–0.00271341 |
| 1024 | reconstruction | 0.00259971 | 0.00256958–0.00262757 |
| 1024 | direct_KL | 0.00254376 | 0.00251778–0.00256174 |

Untouched PCA initialization has mean validation KL.00261506. Larger-pool direct-KL training reaches.00254376, a2.7% mean reduction; each of its three schedules has a lower mean than initialization. Larger-pool reconstruction is approximately tied at.00259971 (0.6% lower). For each schedule and objective, increasing the pool lowers validation mean KL relative to its small-pool counterpart. These are paired descriptive outcomes, not independent dataset replications.

Uncertainty remains important. The paired document-bootstrap interval for larger-pool direct KL minus initialization spans[-.0002204,+.0000465]nats. Larger-pool direct KL versus larger-pool reconstruction also spans zero. The reconstruction coverage contrast excludes zero under the descriptive interval; the direct-KL coverage contrast does not. These10,000-draw intervals average schedules within each document, omit full training-set and schedule uncertainty, and are uncorrected for multiple comparisons. They do not establish a generally better featurizer.

Length-specific direct-KL performance remains uneven. With the larger pool, mean KL at length32 is.000885 versus initialization.001045; at length120 it is.004202 versus initialization.004185. Thus the broader pool largely removes the earlier long-context degradation, while the pooled gain still comes from shorter contexts. Do not turn that observation into a post-hoc choice of evaluation length.

The original H773 schedule and128-document pool reproduce both trained parameter tensors exactly (maximum absolute difference0 for both objectives). This validates repeatability of the training implementation under the expanded runner. Teacher/model parameters remain frozen; document splits are disjoint; source/model/input and saved-fit hashes stay unchanged; all metrics finite; cached-head replay<=5.35e-5. The earlier local numerical gradient audit applies to the unchanged imported training representation and forward implementation. No new claim of exhaustive gradient verification is made.

Compute:13,312 model forwards and6,144context-level KL backward evaluations. Common-training-panel metrics use the same128documents for every condition and are labeled as such, rather than pretending to measure each large pool's full training distribution. The original reserved test is untouched, and the natural document pool remains26,112.

Interpretation: data coverage is a practical limitation of the short pilot, and additional context diversity helps without increasing optimization steps. Direct output fidelity remains a plausible objective, but the observed validation benefit is small and uncertain. Rank-eight compression and lower mean KL alone do not establish stable semantic directions, meaningful manifolds or reusable computational units.

Next: carry all three frozen larger-pool direct-KL fits, their matched reconstruction fits and the unchanged PCA baseline into causal transfer tests. Retain all schedules rather than choose a winner. Test answer-changing and nuisance interventions across competent task families and copying formats; report absolute effect errors where the full intervention is small. This asks whether the modest distribution-level change corresponds to better preservation of computation, instead of assuming that another small KL improvement is sufficient.
