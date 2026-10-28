## H925 — Persistent lexical values dominate value changes, but routing–value interaction remains essential

H924 localized event-order contrast effects to late attention. H925 examines every head at L14 and L16 on all128 transfer rows, retaining both matched contrasts and all task failures. These layers were selected from the preceding scan; this is exploratory decomposition on familiar data, not independent confirmation.

For native routing A and values V, and donor B,W, separate routing-only (B-A)V, mixed-value-only A(W-V), persistent-bus-only, current-value-only, and full donor-head output replacement. The exact local cross term (B-A)(W-V) is computed and audited algebraically but not yet intervened on alone. All edits affect one final-token head output before its projection; the remainder of the model is live.

### Representative largest heads

Heads below are among the strongest full replacements in this exploratory scan; all18 heads are retained in the results. Positive mean drop reduces recipient second-event-name preference.

|Contrast|Head|Replacement|Mean margin drop|RMS effect|Native→edited KL|
|---|---|---|---:|---:|---:|
|event_swap|L14H4|routing|-0.0107|0.1926|0.00196|
|event_swap|L14H4|mixed_values|+0.2437|0.3213|0.00534|
|event_swap|L14H4|bus|+0.2414|0.3191|0.00525|
|event_swap|L14H4|current_values|+0.0023|0.0031|0.00000|
|event_swap|L14H4|full|+0.2317|0.2465|0.00308|
|event_swap|L14H6|routing|+0.0050|0.0987|0.00057|
|event_swap|L14H6|mixed_values|+0.0938|0.1392|0.00108|
|event_swap|L14H6|bus|+0.0912|0.1365|0.00104|
|event_swap|L14H6|current_values|+0.0026|0.0035|0.00000|
|event_swap|L14H6|full|+0.0982|0.1029|0.00060|
|event_swap|L16H3|routing|+0.0110|0.0897|0.00047|
|event_swap|L16H3|mixed_values|+0.0676|0.1131|0.00068|
|event_swap|L16H3|bus|+0.0683|0.1138|0.00068|
|event_swap|L16H3|current_values|-0.0007|0.0012|0.00000|
|event_swap|L16H3|full|+0.0781|0.0802|0.00036|
|same_fact_reword|L14H4|routing|-0.0108|0.2014|0.00217|
|same_fact_reword|L14H4|mixed_values|+0.2438|0.3215|0.00535|
|same_fact_reword|L14H4|bus|+0.2415|0.3193|0.00526|
|same_fact_reword|L14H4|current_values|+0.0023|0.0031|0.00000|
|same_fact_reword|L14H4|full|+0.2317|0.2439|0.00303|
|same_fact_reword|L14H6|routing|+0.0049|0.1009|0.00063|
|same_fact_reword|L14H6|mixed_values|+0.0938|0.1392|0.00109|
|same_fact_reword|L14H6|bus|+0.0912|0.1365|0.00104|
|same_fact_reword|L14H6|current_values|+0.0026|0.0041|0.00000|
|same_fact_reword|L14H6|full|+0.0981|0.1011|0.00061|
|same_fact_reword|L16H3|routing|+0.0110|0.0907|0.00049|
|same_fact_reword|L16H3|mixed_values|+0.0677|0.1133|0.00070|
|same_fact_reword|L16H3|bus|+0.0685|0.1140|0.00070|
|same_fact_reword|L16H3|current_values|-0.0007|0.0014|0.00000|
|same_fact_reword|L16H3|full|+0.0781|0.0794|0.00037|

L14H4 has the strongest individual full replacement effect, followed by L14H6; L16H3 is the strongest at L16. Several other heads contribute, so the whole-layer effect should not be assigned to one head. At L14H4, event-swap full replacement has mean drop0.232 and RMS0.246. Mixed-value replacement has mean0.244, RMS0.321; persistent-bus replacement nearly matches it. Across endpoints their effect RMSE is0.00306. Current-value-only RMS is0.0031. The same pattern appears for same-fact rewording and the other highlighted heads.

Thus persistent lexical values dominate this value-change component. This is consistent with earlier observations, but not sufficient to explain the full operation: routing-only L14H4 has mean effect-0.0107 yet RMS0.193. Averaging signed effects would hide substantial context dependence.

### Nonadditivity is the next discriminating question

At L14H4, the sum of separately measured routing-only and mixed-value-only logit effects differs from the full replacement by RMSE0.381 for event swaps and0.393 for same-fact rewordings. These discrepancies exceed the full replacement's RMS effect. Comparable discrepancies occur at L14H6 and L16H3. Similar mean effects therefore do not establish a faithful decomposition of individual responses.

Two mechanisms can cause this: the explicit local routing-times-value cross term, and nonlinear downstream response when writes are combined. This experiment audits the exact local algebra but does not distinguish their causal importance at the output. Do not conclude routing is dispensable or assign additive percentages to its contribution.

Next intervene on the exact cross term, the simultaneous routing-plus-value write, and their sum at the highlighted heads, retaining both contrasts. Compare a simultaneous combined edit against the sum of separate logit effects to measure downstream nonlinearity. Compare the combined edit with and without the cross term to measure its contextual effect. Verify that adding all three local terms reproduces full donor-head replacement. This directly tests whether the reusable unit is an interaction between a routing coefficient and a lexical vector, rather than either factor alone.

### Verification

23,456 full-model forwards. Identity replay exactly zero; native-cache max error2.05e-5 after batch-size change. Signed attention-times-value reconstruction relative error1.15e-6. Exact routing/value/cross and bus/current decompositions agree to5.93e-14 max absolute error. Source hashes unchanged and all outcomes finite. All-head native captures, endpoint results, local cross-term norms and effect comparisons saved. No reserved-test evaluation or added natural documents; pool remains26,112. Broad goal active.
