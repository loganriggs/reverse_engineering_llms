## H605 — Simple native-context predictors do not restore cross-task transfer

H604's projection modes observed the actual edited-minus-native context difference dc. This experiment removes that information from all predictive modes. Fit dc as a function of native normalized MLP11 input z using the same five leave-task/wording-out folds: 32 other-task old-wording endpoint records for fitting; eight held-out-task new-wording endpoints for evaluation. All predictions still use the actual incoming perturbation e to execute the bilinear response T(predicted_dc,e). The exact/full reference necessarily uses the true dc, but it does not enter predictor fitting or evaluation coefficients.

Compare five shared models: constant mean; isotropic affine change mean_dc + beta*(z-mean_z); centered ridge regression; isotropic change plus ridge residual; and a common shift followed by RMS normalization, predicted_dc=normalize(z+u)-z. Ridge regularization is fixed at 0.1 times the mean training Gram diagonal, without an evaluation sweep. Shift fitting uses 500 Adam steps on training normalized-input MSE only. Same-task old-wording ridge and shift fits are separately labeled diagnostics, not task holdouts. No evaluation-task label is supplied to a shared model, but native context can encode task and format.

The constant shift and normalized-shift models each use one 1,152-dimensional vector. The isotropic affine form adds a scalar and a native-context mean. Ridge uses up to 31 centered training directions plus a vector intercept and many more parameters. These are not equal-capacity comparisons, and a separate dictionary for every task has greater total storage than a single model. The experiment probes prediction forms rather than selecting a smallest representation.

**Predicting from native context does not rescue broad transfer for these models.** Ridge fits training differences much better than a constant shift: relative training error is 0.114–0.199 versus 0.783–0.819. But this advantage largely disappears on the held-out task/new wording. A common shift plus normalization also gives only small, inconsistent benefits. Its training error remains 0.766–0.808. The fitted isotropic contraction beta ranges from -0.0644 to -0.0426; such a small contraction does not explain most of the contextual variation.

| Task | Signal | Mean relative KL | Isotropic relative KL | Ridge relative KL | Isotropic+ridge relative KL | Normalized shift relative KL | Same-task ridge relative KL |
|---|---|---:|---:|---:|---:|---:|---:|
|agreement|number|0.5847|0.4421|0.5775|0.4563|0.4549|0.2851|
|agreement|task|0.6139|0.4942|0.6003|0.5005|0.5044|0.2992|
|agreement|random|0.7107|0.7519|0.6907|0.7432|0.7420|0.6000|
|capital_recall|number|1.0118|0.9915|0.9886|0.9801|0.9911|0.4946|
|capital_recall|task|0.9061|0.8913|0.8711|0.8749|0.8894|0.3647|
|capital_recall|random|0.7914|0.7307|0.8356|0.7775|0.7322|0.3708|
|entity_binding|number|1.1386|1.1670|1.1114|1.0356|1.1581|0.5717|
|entity_binding|task|1.1353|1.2270|1.0206|1.0687|1.2260|0.2840|
|entity_binding|random|1.8924|2.0404|1.7164|1.7669|2.0265|0.9484|
|sentiment|number|1.0060|1.1059|1.0464|1.1574|1.0953|0.1421|
|sentiment|task|1.6096|1.8418|1.7271|1.9992|1.8181|0.1379|
|sentiment|random|1.4564|1.2290|1.4780|1.2707|1.2282|0.1599|
|sequence_copy|number|1.7076|1.6965|1.6377|1.7475|1.7052|0.7728|
|sequence_copy|task|1.1446|1.1473|1.4017|1.3617|1.1475|1.0601|
|sequence_copy|random|2.8723|2.5560|3.1759|2.8363|2.5763|0.7625|

For competent copying, shared ridge relative KL is 1.402–3.176, worse than omitting the context-switch correction; same-task ridge gives 0.763–1.060 and still does not reliably recover the effect. Agreement benefits more from isotropic correction but remains approximate. Same-task sentiment prediction performs better, but its prefixed donors have weak competence, so that is evidence about local response prediction rather than a clean successful-task mechanism.

Relative errors compare each approximation with the exact local context switch, normalized by full-versus-unchanged discrepancy. They are not task error rates. Raw KL, internal response error, answer-margin error and intended-token CE fidelity are stored separately; lower intervention-fidelity error is not necessarily lower task loss. Recipient competence is inherited from H603: copying/agreement fully pass, binding passes with weak prefixed donors, sentiment barely passes with weak donors, recall fails under new wording. No examples are filtered.

An independent training-only L-BFGS refinement checks whether normalized-shift failure is just incomplete Adam fitting. Across shared and same-task fits, it reduces training MSE by at most 0.00178% and changes training predictions by at most 0.276% of the edit norm. The frozen evaluation models are unchanged. This supports near convergence for these fitted solutions; it does not prove a global optimum or a bound on all normalization-based mechanisms.

Updated interpretation: a universal shift or simple contraction/normalization rule at this normalized MLP input does not explain the prefix effect. Linear context prediction can fit the small training sample but generalizes poorly across task and wording together. This favors richer context dependence, or dependence on information not captured by this particular native-input predictor. It does not prove that native context is insufficient in principle, that a larger nonlinear predictor cannot work, or that model features are necessarily task-specific.

Next separate data coverage, structural cues and missing state before increasing predictor flexibility. Use a common sentence shell/terminal format while varying task content, and compare native MLP11 context with earlier residual or attention-derived context as predictors. Include a within-task held-out-item baseline and a within-format held-out-task baseline. These controls can distinguish a task/format distribution shift from missing causal information; merely lowering training error again has little information value.

Audits: actual saved fold IDs verified to exclude evaluation task and wording in main fits; same-task exceptions explicit; source, runner and fitted folds unchanged; 240 cases and 2,160 interventions; native replay max logit difference 3.87e-5 across batching; ridge normal-equation error <=1.42e-15; finite metrics; training-only optimizer audit above. Exact source context-switch algebra was independently checked in H603. No new natural-text documents; FineWeb remains 26,112. Broad goal remains active and incomplete.
