## H938 — Fresh-context transfer reveals a conditional benefit from learned groups

All H935 models remain frozen after discovery-only regularization selection. Evaluate384 H937 endpoints across24 source-vocabulary-disjoint lexical pairs, three domains, four formats and matched copying. All competence cells passed before interventions. A/B reuse training wording with new vocabulary; C/D change wording as well. Replace only persistent-value contributions at three attention readers using live queries; current-value terms stay live. Every seed is retained, with native/donor outputs compared separately.

| Domain/family, format | Continuous effect error | Learned-group seeds | Random-group seeds |
|---|---:|---:|---:|
| Binding task A |8.10%|6.78–8.89%|6.33–8.14%|
| Binding task B |8.30%|7.74–8.77%|7.38–8.65%|
| Binding task C |4.10%|3.65–5.09%|4.03–5.55%|
| Binding task D |14.41%|5.67–9.89%|14.07–17.15%|
| Language task A–D |7.32–9.60%|6.88–9.92%|6.88–9.96%|
| Name copy D |27.93%|13.44–14.55%|22.41–38.33%|
| Country copy D |41.99%|31.48–38.11%|40.92–50.36%|
| Agreement-word copy D |26.78%|10.70–13.01%|20.39–22.29%|

Effect errors are RMS discrepancies in donor-induced answer-logit changes divided by reference-effect RMS. Seed ranges are not confidence intervals. Language task row ranges pool four formats; detailed JSON retains all cells. Agreement task reference effects are tiny, so do not treat their large relative errors as strong architecture evidence.

The strongest result qualifies H936: grouping is not necessary for good familiar-template approximation, but learned context partitions can improve extrapolation to certain new wordings. Binding D uses 'This morning,... passed a note to'; copy D uses repeated 'label'. All three learned-group seeds beat the linear model and corresponding random-group range on these reported D cells. Language-task models remain close, and copying is still imperfect: country-copy D has31–38% error even with learned groups. On country-copy familiarA/B the continuous model remains better than all learned-group seeds. No universal winner or semantic-manifold claim.

Full distributions: binding D nativeKL .0033 continuous vs .0031–.0051 learned, so margin gains do not uniformly improve native output fidelity. Name-copy D nativeKL .0071 continuous vs .0031–.0042 learned. Country-copy D .0200 vs .0173–.0252 remains mixed; agreement-word copy D .0044 vs .0024–.0037 improves. These distinctions preclude treating one metric as total faithfulness.

A posthoc paired lexical bootstrap, averaging squared errors over seeds/endpoints within each of8 pairs, gives pointwise95% learned-minus-linear MSE intervals belowzero for the four D cells above. These are uncorrected descriptive intervals with8 clusters, not independent replicated discoveries; the shared copy template links the three copying results. Full analysis retains A/B/C and all failures rather than selecting D for promotion.

Distance coverage: binding taskD is one token longer than training coverage. The preregistered zero-coefficient fallback affects48 unique example/head positions, or672 prediction calls over7 models and both intervention arms. All predictors use the same fallback; no other cell has missing distances. It may interact with model errors and limits bindingD interpretation. CopyD gains have no missing-distance fallback.

Verification:8064 expected/actual full-model forwards, no-edit identity exact, native logit replay max2.57e-5, source vocabulary disjoint from original discovery/transfer, all finite, tracked hashes including imported predictor and frozen fits unchanged. No refitting or reserved-test evaluation. No new natural documents. Native captures and all perexample metrics retained.

Next priority: compare representational priors directly, per user suggestion. Anchor-state convex hulls, contrast-derived components and unrestricted dictionaries must be separated from shared initialization and anchor-sample choices. First use synthetic known generators where prototype and component hypotheses make different predictions; then reuse verified causal sites and diverse transfer panels. Stability alone is insufficient. Preserve this frozen query comparison as evidence of task-dependent grouping utility, not an invitation to endless template tuning.
