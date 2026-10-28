## H978 — Output-aware weight-derived readouts improve compact composition transfer

Previous turn was progress: H977 separated compact result directions from input-span transfer and found adjective PCA input restriction to be the main bottleneck. H978 freezes the original discovery output and negation bases and derives adjective readouts from the actual bilinear weights. No evaluation scalar calibration, site search or teacher-output coefficient enters the compact prediction.

For unit output direction u define F=L^T diag(Down^T u) R + R^T diag(Down^T u) L. Then the scalar local cross term is a^T F n. Given a frozen negation basis N, store G=F N. Predict u times (a^T G)(N^T n). Output rank1/4 crossed with negation rank1/4 is tested. Rank1 output/rank4 negation requires four adjective linear readouts, four negation readouts, four scalar products and a single output direction; vectors can be correlated and are not claimed independent semantic units. Three random-negation-span controls use the same weight-derived construction.

Inputs a,n remain quartet main-factor vectors, not independently encoded single states. This is therefore a compact operator on factor inputs, not a deployable state-based featurizer. Weight access and supervised discovery factors are explicit priors.

KL ratio measures discrepancy between predicted-local-removal and teacher-local-removal output distributions, normalized to no edit; lower is better.

| Native-passing cohort | PCA input4 | Aware output1/negation1 | Aware output1/negation4 | Aware output4/negation4 | Output1 teacher-projection baseline |
|---|---|---|---|---|---|
| Movie anchors |0.0019|0.0167|0.0053|0.0003|0.0051|
| Book anchors |0.0015|0.0155|0.0065|0.0013|0.0052|
| Book new1 |0.1995|0.0483|0.0251|0.0222|0.0109|
| Book new2 |0.2649|0.1252|0.0617|0.0601|0.0080|
| Book new3 |0.5816|0.2476|0.1789|0.1657|0.0458|
| Film new3 |0.5585|0.2512|0.1861|0.1743|0.0446|
| Movie new3 |0.5741|0.2553|0.1880|0.1784|0.0442|

The original film anchors reproduce the output-projection baseline as expected when the full discovery negation span contains their n factors. Random negation spans yield roughly0.98 normalized KL error. Output-aware readouts materially outperform PCA inputs on new lexicons; four output directions add only modest improvement over one. Remaining gap to privileged output projection shows incomplete coverage of new input factors. Strongest clean transfer case, book/new1, achieves0.0251 with output1/negation4 compared with0.1995 for PCA and0.0109 for teacher-output projection.

This panel was already inspected in preceding experiments; improvement is a comparison on a reused development evaluation set, not fresh confirmation of an untouched test. Native competence gates remain fixed and failed cohorts diagnostic. New2/new3 include multi-subtoken adjectives; last-subtoken alignment remains unchanged.

Audits:3,072/3,072 whole-model forward examples plus192 independent local MLP evaluations. Compact readout contraction equals direct projected bilinear calculation within1.09e-14 relative. Independent direct actual MLP evaluations using f(a+n)-f(a)-f(n)+bias reproduce predicted terms within2.70e-8; H977 comparator margins replay within9.48e-6. Identity logit error2.19e-5; frozen basis/operator hashes unchanged; independent aggregation and finite outputs pass. No gradient training or reserved-test use; natural-document pool26,112.

Interpretation: selecting input directions by their effect on a causally validated result direction is a better prior here than preserving high-variance input directions. This supplies a concrete candidate for a mechanistic featurizer, while retaining a clear gap between factor-based oracle analysis and single-state computation.

Next: build a symmetric single-state quadratic readout using a frozen subspace containing both N and F*N, with discovery-only centering and explicit constant/linear terms. Test scalar output and counterfactual differences from individual normalized activations, without access to evaluation quartet factors at prediction time. Compare with discovery PCA and exact scalar-output controls; then validate promising models on an additional prospective panel. H961 frozen context-bridge predictor evaluation remains pending.
