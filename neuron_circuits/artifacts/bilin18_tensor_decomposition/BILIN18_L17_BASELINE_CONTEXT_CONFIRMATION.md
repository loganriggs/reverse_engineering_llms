## H434 — Independent baseline-context confirmation

Fresh512 complete documents exclude all11456 prior documents by independently rehashed source identity and window comparison. Before intervention outcomes, select16 lowest output-RMS sites and16 nearest the median, all at position63; selected RMS medians950.22 and2645.21. Freeze six natural donor vectors and six independent Gaussian draws, shared across all32 origins and32 output destinations. Factor groups remain H410 core64/added64/union128. No reflection, refitting or response-based context selection. This tests low output RMS as a baseline proxy for the earlier selected case; it is not a randomized causal manipulation of RMS. Total completed unique documents11968.

Native amplitude1 interventions, exact finite KL and baseline Fisher scoring, plus raw reconstruction. Common full-response length is a secondary control, applying the same scale to all group parts. Primary contrasts use original lengths and pooled finite-KL recovery. Joint95% maxabs bands use4000 paired source/document bootstrap samples: source draws paired across natural/Gaussian, and the same sampled selected-document indices reused on origin/destination axes. Intervals condition on the frozen baseline strata and fixed donor pool; they do not resample the entire selection procedure.

| Frozen contrast | Estimate | Joint95% interval | Lower > .05 | Equivalent within±.05 |
|---|---:|---|---|---|
|added_origin_difference_at_median_outputs|+0.501982|[+0.167625, +0.836339]|True|False|
|union_destination_difference_for_median_origins|+0.077908|[-0.256448, +0.412265]|False|False|
|union_origin_destination_interaction|+0.239515|[-0.094842, +0.573872]|False|False|

**One predeclared result confirms:** at median output destinations, added64 recovery is higher for low-RMS origins by50.20 percentage points, joint95% interval[16.76,83.63] points. The >5-point gate passes. Neither the union destination effect nor its origin/destination interaction is established. All equivalence gates fail; unresolved does not mean equivalent.

| Origin | Destination | Core64 KL recovery | Added64 KL recovery | Union128 KL recovery |
|---|---|---:|---:|---:|
|Low RMS|Low RMS|55.20%|11.20%|74.93%|
|Low RMS|Median RMS|68.55%|40.23%|91.09%|
|Median RMS|Low RMS|88.20%|5.43%|90.89%|
|Median RMS|Median RMS|75.57%|-9.97%|83.10%|

The added-group point estimates are+40.23% for low-origin/median-output and-9.97% for median-origin/median-output. The difference is confirmed; the individual signs were not separately tested by the frozen family. Common-length secondary estimates are+34.05% and+9.13%, respectively: origin dependence persists descriptively, but the harmful median-origin sign does not. Do not report a norm-independent sign reversal.

Matched-context results are more demanding than crossed-stratum averages: union128 exact-KL recovery is61.10% in selected low-RMS contexts and84.67% in median contexts. Raw union energy recovery is95.91% versus97.04%. With common response lengths, matched finite-KL recovery is70.61% versus80.03%. These secondary values have no new confidence bands. This limits a claim that128 fixed factors form a sufficient transferable behavioral unit, while preserving their substantial partial coverage.

Numerical gates pass: group partition relative error below frozen1e-5 and KL nonnegativity within1e-10. Independent aggregation reproduces reported pooled recoveries. Original source corpus, frozen group and moment hashes are checked; full-document audit confirms zero overlap. No documents are excluded based on outcomes.

Twenty live explanations after this independent test, with distinguishing observations:

| Hypothesis | Discriminating test |
|---|---|
|Input directions change factor roles|Hold source fixed; compare coefficient-change directions before output scoring.|
|Input RMS gain changes group balance|Separate radial projection from scalar gain under controlled derivatives, then check finite responses.|
|Retained and omitted factors cancel|Measure retained/omitted response cross terms under the same metric.|
|Output directions weight errors differently|Cross response residuals with independent output metrics.|
|Output RMS scale amplifies residual error|Equalize scalar derivative gain while keeping all directions fixed.|
|Likely token identities determine coverage|Swap probability weights with confidence-matched recipients.|
|Probability concentration determines coverage|Match entropy/top-token mass while varying token identities.|
|Finite final normalization matters|Compare exact final RMS with its local tangent using identical responses.|
|Logit capping matters|Keep raw logits fixed and vary cap treatment in diagnostic scoring.|
|Softmax/KL curvature matters|Compare finite KL and quadratic scoring across signed edit sizes.|
|Source norm tails dominate|Common source norms and source-wise influence diagnostics.|
|Source directional tails dominate|Equal norms with fresh directions and paired source sampling.|
|Frozen donor pool is atypical|Repeat with an independently selected donor calibration pool.|
|Empirical moment mismatch matters|Measure conditional moment errors and use matched-moment controls.|
|Position-specific computations matter|Repeat fixed design at other positions with no outcome-based selection.|
|Entity-completion contexts behave differently|Predeclare linguistic categories and test across many documents.|
|Low-RMS selection is only a proxy|Predict recovery from baseline geometry on a held-out panel.|
|Omitted computation is broadly distributed|Compare increasing frozen group budgets on fresh contexts.|
|Native factor structure is important|Compare frozen native groups against matched-size rotation/random controls.|
|Behavioral units require context-conditioned combinations|Predict group membership/weights from baseline state, with held-out behavioral transfer and complexity accounting.|

Priorities: directly inspect retained/omitted error directions and their output weighting before growing the factor budget. The confirmed origin contrast supports changes in response composition; H432 shows receiving-state weighting also matters. Neither result alone distinguishes cancellation from broadly distributed omitted computation. The next high-information test should decompose the fixed group residual under common output metrics across the independent low/median origins, with finite-KL error retained as the behavioral endpoint. If a compact baseline-conditioned correction is proposed, evaluate it on held-out contexts rather than fit the same errors it purports to explain. Semantic categories and model-wide claims remain unestablished.
