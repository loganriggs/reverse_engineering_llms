## H955 — Factorial transfer rejects simple subject-unit versus distractor-residual interpretation

All H949 penalties/seeds and frozen discovery top-eight subsets evaluated on H954's168 complete factorial directed swaps. Both templates pass native competence. Subject-only, distractor-only, and both-number changes are analyzed separately. No fitting or subset changes. 6,048 live model forwards, with exact residual-only/MLP-only/joint and native baselines. H954 supplied336 local teacher evaluations; these are not additional H955 teacher calls.

Full factorial intervention results for joint residual+MLP path. Table spans two seeds. Task error is normalized RMS margin-change error; KLratio is KL(exact intervention || predicted intervention) divided by KL(exact intervention || omitting MLP change). All weights/conditions preserved in JSON; table emphasizes unpenalized full and predesignated moderate-penalty candidates.

| Template: changed factor | Fit | Task error | KL ratio | Absolute KL |
|---|---|---|---|---|
| prepositional:both | lambda=0.0:all64 | 57.344–57.566% | 0.711–0.736 | 0.000621–0.000644 |
| prepositional:both | lambda=0.1:all64 | 42.359–46.518% | 0.685–0.757 | 0.000599–0.000662 |
| prepositional:both | lambda=0.1:top8 | 50.736–56.600% | 0.849–0.875 | 0.000742–0.000765 |
| prepositional:distractor | lambda=0.0:all64 | 64.758–64.957% | 0.627–0.629 | 0.000482–0.000484 |
| prepositional:distractor | lambda=0.1:all64 | 46.079–48.570% | 0.603–0.640 | 0.000464–0.000492 |
| prepositional:distractor | lambda=0.1:top8 | 50.312–54.493% | 0.719–0.753 | 0.000552–0.000579 |
| prepositional:subject | lambda=0.0:all64 | 34.669–36.402% | 0.933–0.937 | 0.000342–0.000344 |
| prepositional:subject | lambda=0.1:all64 | 38.867–44.616% | 0.905–0.966 | 0.000332–0.000354 |
| prepositional:subject | lambda=0.1:top8 | 50.423–58.169% | 0.979–0.998 | 0.000359–0.000366 |
| relative_clause:both | lambda=0.0:all64 | 19.015–19.615% | 0.934–0.942 | 0.000229–0.000231 |
| relative_clause:both | lambda=0.1:all64 | 24.000–24.415% | 0.869–0.877 | 0.000213–0.000215 |
| relative_clause:both | lambda=0.1:top8 | 20.705–24.052% | 0.855–0.883 | 0.000210–0.000217 |
| relative_clause:distractor | lambda=0.0:all64 | 14.806–15.790% | 0.987–1.029 | 0.000107–0.000111 |
| relative_clause:distractor | lambda=0.1:all64 | 17.959–21.639% | 0.916–0.984 | 0.000099–0.000106 |
| relative_clause:distractor | lambda=0.1:top8 | 13.877–26.121% | 0.858–0.914 | 0.000093–0.000099 |
| relative_clause:subject | lambda=0.0:all64 | 17.266–17.926% | 0.736–0.779 | 0.000132–0.000139 |
| relative_clause:subject | lambda=0.1:all64 | 22.329–22.374% | 0.690–0.723 | 0.000123–0.000129 |
| relative_clause:subject | lambda=0.1:top8 | 19.065–20.948% | 0.688–0.773 | 0.000123–0.000138 |

Key discrimination: on competent prepositional subject-only swaps, moderate/top-eight joint error is50–58%, and KLratio.979–.998. Isolated-MLP task error is89–103%, with essentially the same KLratio. Thus these compact components do not simply retain subject-number computation while dropping distractor processing. The unpenalized complete fit also has poor distribution fidelity here (KLratio.933–.937), so the failure is not solely caused by pruning.

On prepositional distractor-only swaps, moderate/top-eight joint error50–55%, isolated-MLP error114–123%, KLratio.719–.753. The exact residual route improves joint task fidelity substantially. A small task margin error on a joint intervention should not be equated with faithful isolated MLP computation.

Relative clauses generalize better by task margin: moderate/top-eight joint errors19–21% subject-only,14–26% distractor-only,21–24% both. Distribution fidelity remains limited: corresponding KLratios.688–.773,.858–.914,.855–.883. The model itself answers all relative-clause endpoints correctly with intended full-vocabulary top1, so teacher task competence is not the explanation for these approximation gaps.

Effect scale matters: exact joint reference RMS is.0906 subject-only/.2124 distractor-only for prepositional, versus.2017/.0801 for relative clauses. Absolute KL values are only about1e-4–8e-4 here. Large relative KL errors characterize failure to reproduce these small intervention distributions, not catastrophic model behavior. The fixed relay captures only part of complete task computation; context may change where number information travels.

Audits: 6,048/6,048 forwards, native identity logits within2.20e-5, exact joint margins within7.29e-6 of H954. Original checkpoint replay exact; fresh product algebra relative error1.89e-6; independent aggregation within4.45e-16; finite records and run-time input hashes unchanged. H954 received audit-only metadata after this run; stripping precisely those added fields recovers the original input hash, whose exact JSON is preserved as bilin18_factorial_number_contexts_at_h955_run.json. No records changed. Reserved test untouched, no refitting, natural document pool26,112.

Interpretation: a compact reproducible response on simple anchors is not a general subject-number unit. Current evidence favors context-restricted approximation, with failures affecting both subject and distractor interventions. It remains unresolved whether the bottleneck is learned input response, missing output directions, or downstream sensitivity to small omitted components.

Next discriminating experiment: evaluation-only output-span oracle. Project each exact teacher MLP finite response onto the frozen model's full64 output span, discovery top8 span, and dominant-product direction. Compare local residuals, then run these oracle changes downstream. Good oracle fidelity with poor actual fidelity implicates input-dependent coefficients; poor oracle fidelity implies the learned output directions cannot represent the missing response. This uses teacher information and is strictly a diagnostic upper bound, not a deployable featurizer or new training result. Context-dependent routing remains a separate possibility if the fixed relay carries little task signal.
