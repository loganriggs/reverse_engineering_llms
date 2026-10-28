## H647 — A discovery-only residual span covers little of fresh context-shift variation

H646 was progress: a local functional eight-gain oracle improved fitted directions modestly but transferred poorly to a withheld direction. H647 tests additional directional structure rather than another scalar calibration.

For each of two task families and two template pairs, concatenate normalized MLP-context changes across layers10–17 into a9216-dimensional vector. Use only the four original H640 discovery items (indices0–3) and subtract the frozen H641 mean shift. Fit SVD directions to those four residual vectors. Because their mean is zero, their full span has rank at most three. Predeclare ranks1,2,3 and keep the same span for reverse template exchange. No output derivatives, probe directions, H640 items4–7 or H642 fresh items enter basis fitting.

For each H642 fresh receiver, compute its true donor-minus-receiver context-shift residual and project it onto the fitted span. Add that correction to the frozen mean shift and propagate the modified late operators. The coefficients are oracle geometric projections using held-out donor contexts, shared across layers and probes. They are not predictions and do not maximize output fidelity. Ranks are reported as a fixed sweep; no winning rank or subset is selected.

**The discovery basis fits its source data exactly but covers little fresh residual variation.** The rank3 span reconstructs discovery residuals within1.68e-15 relative error. On the fresh panel it leaves mean residual-relative norm errors0.951–0.962, capturing only about7–9% of residual energy:

| Fresh condition | Rank1 residual error | Rank2 | Rank3 | Rank3 residual energy captured |
|---|---:|---:|---:|---:|
|Copy matched|0.979|0.966|0.961|7.5%|
|Copy broad|0.974|0.963|0.957|8.2%|
|Fact matched|0.972|0.956|0.951|9.2%|
|Fact broad|0.976|0.966|0.962|7.2%|

These errors concern the residual after removing the already useful mean shift, not the full context change. Energy capture is averaged per example as1−error², rather than computed from the square of the mean error. Exact discovery reconstruction is expected with the complete three-dimensional source span; it is not evidence of generalization.

**Functional improvements are small even with donor-informed coefficients.** Error relative to the exact context-replacement effect is:

| Condition, donor input/readout | Frozen mean | Rank1 correction | Rank2 | Rank3 |
|---|---:|---:|---:|---:|
|Copy matched|0.416|0.414|0.400|0.396|
|Copy broad|0.454|0.435|0.428|0.425|
|Fact matched|0.388|0.372|0.366|0.362|
|Fact broad|0.202|0.200|0.198|0.197|

| Condition, receiver input/readout | Frozen mean | Rank1 correction | Rank2 | Rank3 |
|---|---:|---:|---:|---:|
|Copy matched|0.408|0.404|0.392|0.388|
|Copy broad|0.423|0.403|0.394|0.395|
|Fact matched|0.393|0.373|0.364|0.360|
|Fact broad|0.211|0.209|0.207|0.206|

Rank3 improves207/256 donor-endpoint cases and208/256 receiver-endpoint cases over the frozen mean. Thus there is transferable structure in the small residual span, but its aggregate benefit is modest: donor-condition errors fall0.416→0.396 for matched copying,0.454→0.425 for broad copying,0.388→0.362 for matched facts and0.202→0.197 for broad facts. These improvements use true donor shifts and are not a new predictive result. Geometric projection can slightly worsen functional fidelity in individual cases or as rank rises, because the metrics differ.

**Interpretation must distinguish data coverage, rank and metric.** This test shows that the three directions available from four discovery items do not geometrically represent most fresh residual variation. It does not prove high intrinsic dimension, failure of all low-rank representations, or an information-theoretic limit. The discovery sample is tiny; concatenating all layers imposes coordinated coefficients; SVD optimizes Euclidean context variance rather than response fidelity. Functionally optimized coefficients within the same span could perform differently. Therefore this is a context-space representational test with functional evaluation, not a proven output-error lower bound.

The useful mean operator change survives this result. What remains unrepresented is much of the item-dependent correction around it. The evidence suggests separating a stable shared transformation from less well-covered residual structure, rather than declaring the entire operation arbitrary or claiming a single canonical feature has been isolated.

Next distinguish sample coverage from rank. Use the other four already available H640 items to expand discovery from four to eight while keeping the H642 fresh panel excluded from fitting. Compare the original rank3 span, a rank3 span learned from expanded discovery, and the full expanded residual span (up to rank7). Also compare the original mean with an eight-item mean so that mean estimation and residual coverage are not conflated. Treat this as a retrospective discovery-expansion diagnostic and retain oracle-versus-predictive distinctions. If expanded geometry remains inadequate, test a functionally weighted basis or more varied discovery contexts before training a coefficient predictor.

Audits:512 fresh plus512 discovery cached attention-stage reconstructions;1152 batched JVP calls; no new whole-model forwards. Although all64 discovery-panel native states are reconstructed for consistent batching, only items0–3 enter fitted residual matrices. Frozen mean recovery error1.78e-15; discovery rank3 reconstruction error1.68e-15; all source/runner/basis hashes unchanged; finite metrics. Four fixed basis groups, all ranks, fresh prompts/probes and endpoint combinations retained. No new natural-text documents; FineWeb remains26,112. Existing competence and local-response limitations carry over. Broad goal active and incomplete.
