## H981 — Frozen single-state readout passes a fresh prospective scalar-intervention test

Previous turn was progress: H979 established a single-state output-aware quadratic readout on the development panel. This turn generated H980 after freezing the actual H979 model artifact, then applied that exact artifact unchanged. No new basis fitting, centering, rank choice, scalar calibration, or model-parameter update occurs. New prompts are disjoint from checked preceding panels; twelve adjective pairs are new relative to the declared checked lexicons. All48 quartets are evaluated, with six native-passing book/story cohorts distinguished from failed film/I-thought diagnostics.

The readout uses eight quadratic coordinates plus one linear readout and predicts from a single normalized adjective MLP3 activation. Evaluation swaps scalar-output differences along one frozen output direction u. Reference uses the exact MLP scalar differences; candidate uses differences of its independent single-state predictions. KL ratio is discrepancy to exact scalar-edit outputs divided by no-edit discrepancy. It is not a whole-model accuracy score, full-MLP reconstruction score, or percentage of total task computation.

| Native-passing fresh cohort | Aware8+linear | PCA8+linear | Linear only | Uncentered aware8 |
|---|---|---|---|---|
| Book fresh1 |0.0381|0.3770|0.7436|0.2106|
| Book fresh2 |0.0054|0.2181|0.7393|0.1855|
| Book fresh3 |0.0174|0.2755|0.7357|0.1877|
| Story fresh1 |0.0487|0.4334|0.7712|0.1857|
| Story fresh2 |0.0099|0.2420|0.7305|0.1774|
| Story fresh3 |0.0284|0.2948|0.7453|0.1611|

The output-aware quadratic model consistently beats variance-based and linear controls on this new panel. Fresh2 is a clean allsingle-token cohort with100% all-four binary correctness and75% full-vocabulary top1 in both book and story contexts. Book fresh2 edge-specific KL ratios are0.0033 for negation toggles,0.0065 for adjective changes,0.0174 for both-factor changes; corresponding PCA ratios0.2293,0.2173,0.0937. Thus improvement is not confined to aggregate weighting in this cohort. Other cohorts include multi-subtoken adjectives under the fixed last-subtoken convention.

Native task failures remain visible. Film and I-thought examples often choose the correct side of the binary contrast but fail the full-vocabulary criterion; their fidelity scores are model-response diagnostics rather than successful semantic transfer claims. Lexicon cohorts are shared between contexts, so six passing cohorts are not six independent adjective datasets. The representation is supervised and weight-informed, not an unsupervised manifold or a uniquely identifiable unit.

Audits:4,608/4,608 full-model forward examples, plus H980's193 competence/replay forwards for4,801 total in this prospective test. Native identity logit error2.39e-5. Exact scalar-weight replay max absolute difference0.0160 at teacher scalar RMS6632 (float32 capture versus double algebra). Independent single-example versus batched predictions agree3.64e-12; independent NumPy implementation agrees1.10e-11; summaries independently checked. Original model hash remains unchanged, all outputs finite; no gradient training or original reserved-test access. Natural-document pool26,112.

Interpretation: a causally selected output direction plus weight-aware compact input geometry yields a single-state readout that prospectively predicts scalar counterfactual behavior. This supports a concrete alternative to selecting features solely for activation reconstruction. It does not establish full-layer faithfulness, semantic exclusivity, or that the coordinates correspond to independent natural concepts.

Next: evaluate selectivity and scope. Add same-valence lexical/context changes and non-sentiment native-competent tasks, comparing the frozen scalar channel's effects with full MLP edits. Measure whether the compact readout remains faithful to its channel and which causal effects are necessarily outside it. Preserve model parameters; avoid extending the interpretation from one channel to a general featurizer without those tests. H961 frozen context-bridge predictor evaluation remains pending.
