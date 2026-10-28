## H934 — Continuous corrections help prototypes, but discrete groups are not universally superior

H933 found causal prediction more stable than exact cluster partitions. H934 tests whether discrete grouping adds value relative to a continuous query predictor, and whether within-group variation explains residual errors. This returns to the full96-example discovery split with both templates; evaluate the other96 familiar lexical examples. It is a model-class comparison on that split, not an additional excluded-template or disjoint-sample confirmation.

### Models and correctness checks

Keep the frozen eight-prototype means from H930, all three seeds. Fit a task-blind continuous ridge model from the concatenated normalized pre-rotary queries to routing coefficients, separately by distance from the end. Scale the256-dimensional query by1/16. The continuous model has a fitted unpenalized intercept. The hybrid retains the frozen prototype coefficient and adds a shared linear prediction of its residual from query-minus-assigned-center; no extra residual intercept is fitted. It is not a separately fitted linear model inside each cluster.

Use fixed regularization coefficients.01,.1,1 with objective mean squared coefficient error plus alpha times squared weight norm. Only discovery rows with that position available enter its fit. All settings are retained, without evaluation-based selection. Ridge uses a dual solve; the primal normal-equation residual is audited to6.66e-16. Fit and evaluation indices remain disjoint. Live queries/gates are recomputed through joint interventions at the three fixed readers.

### Native and causal fidelity

Native KL compares the predicted unedited distribution with the model. Edited KL compares donor-value predictions with the native-routing donor intervention. Effect error compares donor-minus-native margin responses relative to reference RMS. Ranges cover all prototype seeds; continuous-only models are deterministic at each regularization value.

|Domain/family|Model / regularization|Native KL|Edited KL|Effect error ratio|
|---|---|---:|---:|---:|
|agreement task|prototype|0.0002–0.0004|0.0003–0.0004|1.9168–2.3178|
|agreement task|linear .01|0.0003|0.0003|1.4688|
|agreement task|linear .1|0.0009|0.0009|1.6325|
|agreement task|linear 1|0.0029|0.0028|1.7404|
|agreement task|hybrid .01|0.0002–0.0002|0.0002–0.0002|1.7576–2.1699|
|agreement task|hybrid .1|0.0002–0.0003|0.0003–0.0003|1.8320–2.3027|
|agreement task|hybrid 1|0.0002–0.0004|0.0003–0.0004|1.9051–2.3165|
|agreement copy|prototype|0.0093–0.0103|0.0081–0.0090|0.2780–0.3098|
|agreement copy|linear .01|0.0040|0.0038|0.2112|
|agreement copy|linear .1|0.0053|0.0051|0.2884|
|agreement copy|linear 1|0.0062|0.0065|0.4269|
|agreement copy|hybrid .01|0.0055–0.0063|0.0050–0.0058|0.2416–0.2717|
|agreement copy|hybrid .1|0.0085–0.0095|0.0074–0.0083|0.2699–0.3028|
|agreement copy|hybrid 1|0.0092–0.0102|0.0080–0.0089|0.2770–0.3091|
|language task|prototype|0.0312–0.0329|0.0447–0.0467|0.1588–0.1630|
|language task|linear .01|0.0523|0.0911|0.2605|
|language task|linear .1|0.2231|0.3602|0.5567|
|language task|linear 1|0.5966|0.7436|0.8280|
|language task|hybrid .01|0.0297–0.0315|0.0430–0.0449|0.1565–0.1605|
|language task|hybrid .1|0.0310–0.0327|0.0445–0.0465|0.1584–0.1627|
|language task|hybrid 1|0.0312–0.0329|0.0447–0.0467|0.1587–0.1630|
|language copy|prototype|0.0266–0.0300|0.0361–0.0464|0.4344–0.4936|
|language copy|linear .01|0.0142|0.0369|0.3736|
|language copy|linear .1|0.0474|0.0739|0.4589|
|language copy|linear 1|0.0744|0.1218|0.5360|
|language copy|hybrid .01|0.0192–0.0207|0.0280–0.0332|0.3783–0.4184|
|language copy|hybrid .1|0.0250–0.0276|0.0344–0.0431|0.4230–0.4762|
|language copy|hybrid 1|0.0264–0.0297|0.0359–0.0460|0.4332–0.4916|
|binding task|prototype|0.0117–0.0144|0.0114–0.0137|0.0810–0.0852|
|binding task|linear .01|0.0068|0.0043|0.1397|
|binding task|linear .1|0.0234|0.0254|0.4651|
|binding task|linear 1|0.0469|0.0774|0.7609|
|binding task|hybrid .01|0.0056–0.0057|0.0050–0.0051|0.0719–0.0735|
|binding task|hybrid .1|0.0094–0.0108|0.0092–0.0105|0.0791–0.0827|
|binding task|hybrid 1|0.0114–0.0139|0.0111–0.0132|0.0808–0.0849|
|binding copy|prototype|0.0061–0.0066|0.0018–0.0026|0.0998–0.1049|
|binding copy|linear .01|0.0081|0.0032|0.1583|
|binding copy|linear .1|0.0282|0.0177|0.4008|
|binding copy|linear 1|0.0501|0.0465|0.6138|
|binding copy|hybrid .01|0.0051–0.0052|0.0017–0.0019|0.0928–0.0943|
|binding copy|hybrid .1|0.0059–0.0063|0.0018–0.0024|0.0986–0.1020|
|binding copy|hybrid 1|0.0061–0.0066|0.0018–0.0025|0.0997–0.1045|

For binding task, hybrid.01 improves native KL from prototype0.0117–0.0144 to0.0056–0.0057 and effect error from8.1–8.5% to7.2–7.4%. Binding copying also improves modestly. Language task gains are smaller; prototypes and hybrids outperform the tested continuous-only ridge settings in effect fidelity.

However, continuous-only.01 does better on agreement copying and language copying than the prototype baseline, and can outperform the hybrid on those cases. For example, language copying has37.4% effect error and native KL0.0142 under continuous-only.01, versus43.4–49.4% and0.0266–0.0300 for prototype means. There is no universal winner across contexts. Agreement task retains the tiny-reference-effect caveat from prior experiments.

### Interpretation limits and next experiment

A discrete context baseline plus within-group continuous variation is useful for some pathways. This is compatible with locally structured computation, but does not prove semantic manifolds or that discrete groups are fundamental. The hybrid has additional structure/capacity; a strong continuous-only baseline must receive fair model selection. All strongest results are at the weakest regularization tested (.01), so this grid has not located an interior optimum. No claim of optimized model-class superiority is warranted.

Next choose regularization using discovery-only lexical cross-validation, expanding the predeclared range toward weaker penalties, then evaluate frozen choices once on the existing research-transfer panel. Add a comparable prototype/correction control with nonsemantic grouping (for example a fixed random partition/gate) to test whether improvements require meaningful query grouping rather than extra capacity. Keep native distribution and intervention fidelity as separate outcomes. After model choice is fixed on discovery data, repeat excluded-template and disjoint-sample checks rather than tuning on their failures.

These comparisons are exploratory on a repeatedly used evaluation panel. A genuinely new task/template panel is still needed for strong generalization claims. The broad objective remains identifying reliable computational units, not maximizing performance on these examples.

### Verification

3,744 full-model forwards. Identity replay exactly zero, native replay within1e-4, source hashes unchanged and all metrics finite. Shared H931 baselines—including all three prototype models—replay exactly. Ridge normal equations pass and fitting uses discovery examples only. Every fixed regularization setting, seed, regression weight and endpoint metric saved. No reserved test or added natural documents; pool remains26,112. Broad goal active.
