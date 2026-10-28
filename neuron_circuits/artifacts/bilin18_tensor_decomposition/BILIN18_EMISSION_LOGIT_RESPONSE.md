## H624 — Small-dose vocabulary responses depend on context beyond a shared scalar gain

The previous turn was progress: H623 found a sign reversal for language prompts and exposed incomplete plural candidate sets. H624 measures the full-vocabulary logit response to the same frozen final Attention5 vector at addition-signed strengths−1,−0.25,−0.05,−0.02,0,0.02,0.05,0.25,1 on72 prompts:648 sequence evaluations. Central differences at0.02 estimate a local response, checked against0.05 differences.

Fit response vectors using predeclared first-four items: (a) mean response from capital form0 only, (b) pooled mean across all nine relation/form cells, and (c) a separate mean per cell. Evaluate the remaining four items per cell. An oracle scalar fit rescales the capital response separately for each evaluation prompt; it is a diagnostic, not a deployable predictor. Each prompt's own local response is a second oracle, measuring finite-dose linearization error. The original residual intervention vector is never refitted.

**A fixed capital vocabulary response does not explain other contexts well, even with oracle gain adjustment.** On held-out capital form0 the probability-weighted centered response cosine is0.971 and relative vector error0.265. On language forms, cosines with that capital response are−0.073,−0.181 and−0.522. Even allowing a signed scalar gain leaves0.85–0.97 relative response error on language and0.91–0.95 on plurals. Thus different overall strength or sign is not enough to reproduce the response shape on these prompts.

Errors are measured after removing the common logit offset, with each recipient's native vocabulary probabilities as weights. These are local distribution-sensitive vector metrics. Cross-context cosine under this weighting is not a basis-invariant semantic angle or a statement about hidden feature identity.

| Held-out cell | Capital-response cosine | Capital vector error | Oracle scalar error | Per-cell mean error | Actual intended-token log-prob derivative | Capital-response prediction |
|---|---:|---:|---:|---:|---:|---:|
|capital/0|0.9711|0.2650|0.2357|0.2650|0.5447|0.5701|
|capital/1|0.8209|0.6299|0.5563|0.5480|2.8758|1.4925|
|capital/2|0.4051|1.4518|0.8961|0.4223|0.4922|0.7233|
|language/0|-0.0727|1.7108|0.9484|0.6026|0.1149|-0.1240|
|language/1|-0.1807|1.9019|0.9748|0.3007|-0.2353|-0.0582|
|language/2|-0.5216|1.7952|0.8506|0.2928|0.0904|-0.1227|
|plural/0|0.3003|1.1904|0.9460|0.5094|0.2220|-0.3195|
|plural/1|0.4017|1.3051|0.9122|0.2873|0.3563|-0.3908|
|plural/2|-0.3344|3.1714|0.9193|0.6558|0.1178|-0.5857|

The “speak” construction's negative intended-language derivative is already present locally (mean−0.235 on held-out items), not only at the large intervention strength. Capital-derived fixed bias predicts−0.058 there, but misses the overall response shape and predicts the wrong sign for language forms0/2. The capital mean's strongest raw logit increases include city tokens such as Copenhagen, Hamburg, London, Havana and Paris. That is a descriptive signature in its training context, not evidence that the injected residual vector intrinsically encodes cities in every context.

**The result is not just a poor averaging objective.** As an adaptive offline audit, fit one shared response using exact native-probability-weighted least squares, including a free common-logit offset per training example. The normal equations are solved on the same declared training indices; maximum residual is below7.2e-16. Probability weighting improves some pooled predictions but does not produce uniform transfer. On held-out language “speak”, weighted pooled response has relative small-dose KL1.93; a per-cell mean has0.093. Capital-only weighted response has3.83, and its oracle scalar response error remains0.968. These fits are specific candidate models, not a proof that no useful common component exists.

| Held-out cell | Capital mean relative KL | Pooled mean relative KL | Weighted pooled relative KL | Per-cell mean relative KL | Own tangent relative KL |
|---|---:|---:|---:|---:|---:|
|capital/0|0.0782|0.4369|0.1654|0.0782|0.0009|
|capital/1|0.3650|0.6895|0.3778|0.2718|0.0002|
|capital/2|1.9182|0.6914|0.8803|0.1686|0.0010|
|language/0|3.0405|0.8937|1.4417|0.3976|0.0013|
|language/1|3.5948|2.0009|1.9300|0.0930|0.0007|
|language/2|3.1576|0.9764|0.9016|0.1236|0.0006|
|plural/0|1.3553|0.8120|0.5202|0.2686|0.0011|
|plural/1|1.7806|0.6991|0.4458|0.0964|0.0022|
|plural/2|9.7829|1.8909|1.6894|0.4307|0.0010|

Relative KL here compares the actual+0.05 distribution with a predicted distribution native_logits+0.05×response, normalized by predicting no change. Values above1 are worse than omission. Per-cell fits generally improve prediction, but retain material error and have more parameters plus known cell labels. Pooled and per-cell discovery include all forms and related subject identities; only item IDs4–7 are excluded. This is not an independent new-task evaluation.

**Local numerical estimates are stable, while large-dose linearization fails.** Held-out0.02-versus0.05 derivative discrepancies are about0.0003–0.0005 in relative weighted norm. Each prompt's own tangent predicts+0.05 with relative KL0.0002–0.0022. At addition/subtraction strength1, own-tangent relative KL ranges roughly0.087–1.218 across cells/signs. Two issues therefore coexist: context changes the local response, and finite-strength curvature changes its extrapolation. A local shared response and a large-dose constant response are different hypotheses.

**The plural competence audit is now corrected.** Add each singular form to the candidate set rather than comparing only eight plural words. Native choice drops from100% among the plural-only set to0% among plural plus singular candidates on every tested plural prompt. This independently corroborates the short-generation audit: previous high candidate choice did not demonstrate pluralization. It remains valid to study distributional effects on these prompts, but not to call them successful plural-task preservation.

Mechanistic implication: the fixed hidden direction is transformed into a context-dependent logit effect. This does not yet locate where dependence arises. Final RMS normalization and logit saturation are themselves context-dependent, so failure of a shared logit vector cannot be attributed specifically to intermediate attention or MLP computation. A common pre-readout residual response could still produce different logit responses through the same context-dependent final readout.

Next capture response vectors through downstream layers and analytically separate final RMS-normalization and tanh derivatives. Compare a shared pre-readout residual response passed through each recipient's native readout with the observed logit tangent. If that explains transfer, the main dependence is in readout; if not, locate where intermediate response trajectories diverge. This tests a concrete mechanistic alternative rather than merely fitting another output basis.

Audits: source/manifest/vector/runner hashes unchanged; all648 evaluations finite; first4 discovery IDs and36 pooled discovery examples recorded before evaluation; response fits frozen for scoring; derivative step-size check passes; weighted audit satisfies normal equations and uses the same exclusions. Response captures and fits saved. Full-vocabulary errors concern this intervention and data distribution, not universal feature equivalence. FineWeb remains26,112 natural-text documents. Broad research goal active and incomplete.
