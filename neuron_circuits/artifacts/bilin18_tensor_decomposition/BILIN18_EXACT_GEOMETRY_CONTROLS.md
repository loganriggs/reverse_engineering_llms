## H716 — Exact decoder geometry controls retain a small context-dependent response advantage

Previous goal turn classification: progress. H715 measured4572model responses and identified decoder-geometry mismatch as a remaining confound. H716 removes that specific mismatch using rotations and directly checks the linear predictions.

Take the134Dspan of all measured unit decoder directions. For each frozen group hold its anchor fixed. Decompose the seven neighbors into their anchor-parallel components and a seven-dimensional orthogonal residual frame. Replace that residual frame with a random orthonormal frame in the anchor-perpendicular part of the measured span, keeping its coefficient matrix unchanged. This preserves every neighbor norm, every anchor-neighbor inner product and every neighbor-neighbor inner product. Generate199joint control panels with seed716, each containing one independent rotation for each ofall8groups; reuse each panel acrossall16H715contexts. Selection never uses output scores.

Use the saved symmetric derivatives to predict each rotated direction's Fisher-weighted output response, with and without the local temperature direction removed. Compare real versus rotated mean squared anchor-neighbor response cosines.

| Metric | Real groups | Mean exact-geometry control | Difference | Bootstrap95%over contexts | Control panels at least as large, of199 |
|---|---:|---:|---:|---|---:|
|fisher|0.30880|0.28803|0.02077|[-0.00597,0.05029]|1|
|confidence_removed|0.27866|0.26298|0.01568|[-0.00968,0.04345]|5|

The two uncertainty calculations answer different questions. Conditional on these16contexts, only1of199rotated panels reaches the real full-response score and5reach the confidence-removed score; add-one upper-tail estimates are0.01and0.03. These are conditional rotation-null estimates, not universal significance claims. Resampling contexts gives intervals including zero for both contrasts. Only9of16contexts have positive differences. Thus there is evidence of alignment beyond exact decoder geometry on this particular panel, but broad context generalization remains unestablished.

The confidence-removed group differences range-0.02149to+0.04609. Groups7and200are negative;216and184are among the larger positives. These descriptive differences do not authorize winner selection or semantic labels. The whole-group average remains the frozen contrast.

The rotations address residual-space geometry but deliberately do not preserve natural activation statistics, sparse support, lexical associations or on-data status. They complement H714's activation-statistic matches rather than replace them. The control span is134D, not the entire1152Dresidual space. A computational interpretation still requires selective task effects and transfer across contexts/combinations.

### Direct implementation and approximation checks

Decoder Gram preservation maximum absolute error is2.00e-15. The measured decoder matrix has condition number7.33, so solving for rotated directions is not near-singular. Recomputed real response scores reproduce H715within2.92e-8.

Before examining scores, fix16verification directions: first rotated control's first neighbor for each of8groups in the first2contexts. Run positive/negative1%residual-norm model interventions for each,32forwards total. Actual versus predicted Fisher response signatures have relative error median0.000889and maximum0.001579; minimum cosine0.99999935. This supports using the local derivative model for these rotations at this scale. It does not validate arbitrary large edits or every possible context.

### Next research step

The geometry objection alone does not erase the small response alignment, but increasingly precise context-free similarity measurements are unlikely to identify what the groups compute. Move to the existing diverse paired-task framework: entity binding, agreement, sequence copying, capital recall, sentiment, negation, relational order and context association. Recheck native competence and report ambiguous/failed families rather than interpreting their intervention scores as successful computation. Use predefined answer contrasts, nuisance changes and discovery/transfer templates; keep all eight candidate groups, scalar-anchor baselines and controls. Measure whether a group's effect tracks the required answer change and transfers across wording, rather than merely aligning output derivatives.

This is a falsification/interpretation test of provisional candidates, not a declaration that a superior featurizer or manifold has been verified. For families where the model cannot perform the task, retain that outcome and avoid assigning mechanisms from prompted labels. Any group-specific discovery requires independent transfer evidence. Original corpus reserved-test data remain untouched.

All model/source/input hashes unchanged, all scores finite.32additional whole-model forwards; no new unique research documents, pool26112. Broad goal active.
