## H602 — Fresh context edits break much of the shared basis's transfer

Freeze H601's broad 64-dimensional basis and both fitted response metrics before new inference. Construct five tasks with eight pairs each, using familiar lexical inventories but new wording: package-arrival copying, museum/ticket binding, close-to-the-gate agreement, city-that-serves-as-capital recall, and concert-felt sentiment. For each endpoint create a same-answer context by prefixing "For the record,". Also use the opposite-answer endpoint as a separate context donor. This gives 160 native prompts with zero exact string overlap against the checked H581/H582 manifests. It does not establish novel vocabulary or training-data novelty.

Cross context type with number/task/random Attention5 signals at the same norm and both signs. Keep the random direction from H600 so wording/context transfer is not confounded with a new random vector. Capture 960 runs (480 distinct recipient/signal/sign conditions repeated for the two context types), then execute 5,760 downstream interventions: full, unchanged, Euclidean projection, number-trained metric, multi-signal metric and oracle. No basis or map fitting occurs.

Agreement and copying pass competence with full-vocabulary intended-group top1 1.000. Sentiment passes at 0.5625; its prefixed nuisance variants fall to 0.3125. Recall and binding both fail with full-vocabulary top1 0.000 despite correct candidate contrasts. All examples remain. Findings on recall and binding cannot be treated as successful-task mechanisms. The copying and agreement failures below do not depend on those competence failures.

The broad basis loses much of its fidelity. In competent copying, number-metric relative KL is 0.722–1.061 for prefixed contexts and 0.819–0.888 for answer-changing contexts. Relative internal errors are about 0.905–0.921. These are far worse than the familiar-context H601 results, where broad number-metric KL was 0.018–0.076 across the panel. These comparisons characterize a distribution shift; wording, edit type and item composition changed, so they do not isolate which change caused it.

Agreement is a useful exception with limits: answer-changing contexts under number/task signals retain relative KL around 0.074–0.076, but margin errors remain about 0.264–0.266 and internal errors 0.647–0.665. Under random signals the same context class has relative KL 0.261. Good output fidelity can coexist with large local response error.

| Task | Context | Signal | Gate | Number-metric internal error | Number-metric relative KL | Multi-metric relative KL | Oracle internal error |
|---|---|---|---|---:|---:|---:|---:|
|agreement|nuisance|number|pass|0.8537|0.3037|0.3288|0.8036|
|agreement|nuisance|task|pass|0.8585|0.3299|0.3508|0.8097|
|agreement|nuisance|random|pass|0.8933|0.7201|0.7102|0.8503|
|agreement|answer_change|number|pass|0.6474|0.0740|0.1303|0.5258|
|agreement|answer_change|task|pass|0.6651|0.0762|0.1324|0.5427|
|agreement|answer_change|random|pass|0.9071|0.2614|0.2900|0.7844|
|sequence_copy|nuisance|number|pass|0.9054|0.7535|0.7615|0.8670|
|sequence_copy|nuisance|task|pass|0.9209|0.7216|0.7365|0.8756|
|sequence_copy|nuisance|random|pass|0.9183|1.0613|1.0389|0.8796|
|sequence_copy|answer_change|number|pass|0.9203|0.8411|0.8274|0.8648|
|sequence_copy|answer_change|task|pass|0.9159|0.8880|0.8757|0.8586|
|sequence_copy|answer_change|random|pass|0.9159|0.8189|0.8236|0.8688|
|capital_recall|nuisance|number|fail|0.7052|0.4474|0.4551|0.6825|
|capital_recall|nuisance|task|fail|0.7334|0.6987|0.6944|0.6960|
|capital_recall|nuisance|random|fail|0.7533|0.5335|0.5170|0.7170|
|capital_recall|answer_change|number|fail|0.9216|0.7676|0.7690|0.8860|
|capital_recall|answer_change|task|fail|0.9311|0.8651|0.8665|0.8877|
|capital_recall|answer_change|random|fail|0.9300|0.7768|0.7610|0.8830|
|entity_binding|nuisance|number|fail|0.9108|0.7670|0.7660|0.8567|
|entity_binding|nuisance|task|fail|0.8853|0.7393|0.7202|0.8417|
|entity_binding|nuisance|random|fail|0.9106|0.7806|0.7886|0.8512|
|entity_binding|answer_change|number|fail|0.8936|0.7633|0.7285|0.7952|
|entity_binding|answer_change|task|fail|0.9099|0.7514|0.7543|0.8448|
|entity_binding|answer_change|random|fail|0.9274|0.7814|0.7631|0.8784|
|sentiment|nuisance|number|pass|0.8792|0.6555|0.6529|0.8341|
|sentiment|nuisance|task|pass|0.8796|0.7589|0.7428|0.8324|
|sentiment|nuisance|random|pass|0.9078|0.8737|0.8971|0.8682|
|sentiment|answer_change|number|pass|0.8334|0.7612|0.7537|0.7966|
|sentiment|answer_change|task|pass|0.7723|0.5370|0.4484|0.6497|
|sentiment|answer_change|random|pass|0.8415|0.6449|0.6216|0.7943|

All relative errors concern fidelity to a particular local MLP11 context-switch correction, with the original upstream signal and downstream network live. They do not measure task accuracy or the fraction of total computation explained. For answer-changing context donors, only the bilinear response's context factor is switched; this is not a complete counterfactual prompt or donor computation. Intended-token CE remains measured against the recipient's explicit answer, with group NLL kept separate in raw records.

The per-case oracle leaves large local errors too, including about 0.859–0.880 for copying. Thus no coefficient rule can recover the missing internal response exactly within this frozen span. This is not an impossibility result for another basis, a larger representation, or output KL/CE optimization.

Hypothesis update: H601 demonstrated shared structure across familiar nuisance edits and formats, but did not establish general task units. H602 supports the narrower explanation that the span is strongly conditional on the context-change distribution. It weakens a broadly reusable 64-dimensional context representation. It does not refute the idea of data-dependent equivalence; it shows that the data distribution defining that equivalence must include the relevant kinds of change. Agreement's retained answer-change output effect suggests some specific signal-dependent structure survives.

Next isolate the shift: cross old/new wording with old/new nuisance-edit type, keeping lexical items and incoming signals matched. Copying and agreement provide competent anchors. Freeze the current maps; use per-case oracles to distinguish span failure from coefficient failure. Separately test rank/coverage expansion only after this diagnostic, so additional fitting does not obscure the source of failure. The ultimate unit-of-computation question remains open: a useful coordinate span is not itself evidence for uniquely identifiable semantic units.

Audits: 160 native prompts, 960 capture forwards, 5,760 interventions; no exact overlap with checked manifests, no refit, frozen sources/maps and manifest unchanged; exact native logit replay; equal upstream norms within 1.2e-5; oracle normal-equation error <=1.43e-15. The float32 finite-difference audit shows cancellation up to 0.00109 relative error. An independent float64 audit across all 960 records matches saved cross terms within 4.19e-7 and verifies the identity within 6.01e-13. All metrics finite. No new natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
