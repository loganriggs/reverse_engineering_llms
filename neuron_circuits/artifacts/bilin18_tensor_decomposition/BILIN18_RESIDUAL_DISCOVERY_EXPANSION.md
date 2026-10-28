## H648 — More discovery coverage improves a fixed-rank residual representation

H647 was progress: a three-direction span from four discovery items captured little fresh residual structure, but left open whether the bottleneck was data coverage, rank or the geometric objective. H648 separates coverage and rank while keeping the fresh panel out of fitting.

Expand discovery to all eight original H640 items per family. These include items4–7 previously used as H641 retrospective holdouts; they are now explicitly discovery data. The H64264-prompt panel and four probes remain excluded from fitting. Because that panel has already been examined, this is a retrospective discovery-expansion diagnostic, not a new blind replication.

Fit an eight-item mean and centered residual SVD for each family/template pair, concatenating layers10–17. Compare: the original four-item mean; its original rank3 residual correction; an eight-item-trained rank3 span with the old mean; the expanded full rank7 span; and the new mean alone or with those expanded spans. Every residual-basis correction uses oracle Euclidean projection of the true fresh donor shift. Mean-only models need no donor shift at evaluation. No basis rank or favorable subset is selected.

**Coverage matters at fixed rank.** Under donor incoming/readout, the old-mean plus rank3 effect error falls0.396→0.330 for matched copying and0.425→0.382 for broad copying when the basis is learned from eight items instead of four. Fact improvements are smaller,0.362→0.354 and0.197→0.190. Thus H647's limited residual transfer was partly a discovery-coverage problem; it was not evidence that three directions could never capture useful additional structure.

| Condition, donor input/readout | Old mean | Original rank3 | Expanded rank3, old mean | Expanded rank7, old mean | New mean only | Expanded rank3, new mean |
|---|---:|---:|---:|---:|---:|---:|
|Copy matched|0.416|0.396|0.330|0.317|0.371|0.326|
|Copy broad|0.454|0.425|0.382|0.368|0.419|0.379|
|Fact matched|0.388|0.362|0.354|0.340|0.374|0.351|
|Fact broad|0.202|0.197|0.190|0.188|0.198|0.190|

Receiver-input/readout comparison:

| Condition | Old mean | Original rank3 | Expanded rank3, old mean | Expanded rank7, old mean | New mean only | Expanded rank3, new mean |
|---|---:|---:|---:|---:|---:|---:|
|Copy matched|0.408|0.388|0.324|0.312|0.365|0.320|
|Copy broad|0.423|0.395|0.347|0.340|0.388|0.345|
|Fact matched|0.393|0.360|0.352|0.338|0.372|0.348|
|Fact broad|0.211|0.206|0.201|0.192|0.204|0.195|

**Increasing rank provides an additional, usually smaller gain.** With the old mean fixed, expanded rank3→rank7 errors fall0.330→0.317 and0.382→0.368 for copying,0.354→0.340 and0.190→0.188 for facts. These observed comparisons do not establish an optimal rank or a general learning curve from only two discovery sizes. Substantial error remains even under oracle reconstruction.

**Better mean estimation also helps without oracle coefficients.** The new eight-item mean reduces donor-condition errors from0.416→0.371 and0.454→0.419 for copying,0.388→0.374 and0.202→0.198 for facts. These are actual mean-model predictions of the component effect, although evaluated retrospectively on an already studied fresh panel. The directional corrections in the other columns are still oracle results. The original frozen H641 artifact remains unchanged; the new means are stored separately.

Geometric residual coverage improves too. Using the old-mean residual as a common denominator, explained residual energy rises from H647's7–9% to:

| Condition | Expanded rank3, old mean | Expanded rank7, old mean | New mean only | Expanded rank3, new mean |
|---|---:|---:|---:|---:|
|Copy matched|30.2%|34.1%|19.4%|31.8%|
|Copy broad|22.1%|26.4%|12.9%|23.3%|
|Fact matched|14.1%|19.6%|8.0%|17.1%|
|Fact broad|11.6%|17.5%|6.8%|15.1%|

For the mean-only column, this is reduction in squared residual error relative to the old mean, not projection energy onto a subspace. The shared denominator makes comparisons possible but the quantities are not causal shares. Geometry and functional error need not improve in identical proportions.

At full expanded rank7, choosing the old or new mean produces the same affine reconstruction to1.78e-15 absolute error: their difference lies inside the complete discovery residual span. This is a useful algebraic control and explains why the two rank7 functional results coincide. Rank3 spans lack that guarantee, so changing the base mean can still matter. The expanded rank7 span reconstructs its eight discovery residuals within1.07e-15 relative error, as expected; fresh transfer is the substantive test.

The updated picture is a stable shared transformation plus residual variation for which discovery coverage is important. Increasing sample diversity improves both a simple mean predictor and an equal-rank correction span, especially for copying. This weakens an interpretation that the fresh residual is merely unstructured or intrinsically too high-dimensional to model. It still does not establish that rank3 or7 is sufficient for high-fidelity explanation, that the best output-metric representation has been found, or that its coefficients can be predicted from the receiver.

Next test coefficient prediction separately from oracle capacity. Use the eight discovery items to map receiver-native context features to rank3 correction coefficients, keeping each task/template condition explicit. Use a fixed, declared regularization rule or genuinely nested discovery-only validation; do not tune on the fresh panel or leak held-out targets through basis fitting during validation. Compare predicted coefficients with the eight-item mean and oracle rank3 correction, retaining all input/readout conditions and probe types. This determines whether the additional representable structure is accessible from the receiver rather than only from the donor target.

Audits:512 fresh plus512 discovery cached attention evaluations;1920 batched JVP calls; no new whole-model forwards; old mean recovery within1.78e-15; expanded rank7 source reconstruction within1.07e-15; old/new-mean affine equivalence within1.78e-15; all source/runner/basis hashes unchanged; finite metrics. All64 fresh prompts and four probes retained. No new natural-text documents; FineWeb remains26,112. Existing local-response and task-competence limits remain. Broad goal active and incomplete.
