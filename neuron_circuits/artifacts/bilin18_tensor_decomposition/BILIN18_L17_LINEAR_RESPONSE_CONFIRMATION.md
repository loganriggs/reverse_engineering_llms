

## H390 — Fresh-document confirmation of input-derived first-order prediction

Predict the derivative at t=0 of RMS(pre + B(RMS(pre−t·payload), RMS(pre−t·payload)) + Down_bias). This holds the direct residual pre fixed and matches the existing MLP-only source-removal target. Differentiate input RMS, the bilinear map, and final RMS analytically, using the original float32 RMS epsilon. Prediction uses intact residual, source payload and native weights; it never reads the exact finite response. Use512 fresh documents excluding all5,824 prior panel document hashes, with new moment seeds and donor shifts. Heads0/1/6, four positions and all39 response conditions remain fixed.

Fit ranks16/64 in the frozen probability/random bases to the predicted normalized response using the recipient probability metric without cap derivative. Compare four paths: freshly recomputed exact-response coefficient/step oracle; linear coefficients with true-target step oracle; linear coefficients with step selected solely against the predicted full response; and linear coefficients at unit step. The predicted-target step uses the same eight-step grid and full capped-vocabulary scoring, with the predicted distribution as its reference. Also score the unrestricted linear prediction. Exact finite targets are used only for evaluation and the explicitly labeled oracle controls. No second RMS is applied after the predicted normalized displacement. This removes exact finite target access from the practical path, but retains native weights and the known intervention payload: it is not parameter compression or an unconditional next-token predictor.

|Basis|Head|Family|Exact oracle64|Linear true-step64|Linear predicted-step64|Linear unit-step64|Full linear|
|---|---|---|---:|---:|---:|---:|---:|
|probability|0|actual|0.8523|0.8459|0.8446|0.8446|0.9925|
|probability|0|moment|0.7327|0.7221|0.7203|0.7204|0.9838|
|probability|0|donor|0.7397|0.7294|0.7274|0.7274|0.9839|
|probability|1|actual|0.7916|0.7866|0.7862|0.7859|0.9887|
|probability|1|moment|0.7773|0.7714|0.7705|0.7704|0.9856|
|probability|1|donor|0.7816|0.7757|0.7750|0.7750|0.9865|
|probability|6|actual|0.8158|0.8019|0.7865|0.7842|0.9672|
|probability|6|moment|0.8485|0.8460|0.8319|0.8185|0.9678|
|probability|6|donor|0.8515|0.8499|0.8363|0.8060|0.9676|
|random|0|actual|0.8019|0.7947|0.7930|0.7928|0.9925|
|random|0|moment|0.7127|0.7024|0.7001|0.7002|0.9838|
|random|0|donor|0.7172|0.7070|0.7045|0.7046|0.9839|
|random|1|actual|0.7618|0.7579|0.7574|0.7571|0.9887|
|random|1|moment|0.7456|0.7413|0.7402|0.7403|0.9856|
|random|1|donor|0.7505|0.7462|0.7454|0.7454|0.9865|
|random|6|actual|0.7670|0.7551|0.7378|0.7362|0.9672|
|random|6|moment|0.8241|0.8212|0.8055|0.8114|0.9678|
|random|6|donor|0.8274|0.8251|0.8086|0.8151|0.9676|

Frozen gates: {'predicted_fit_within5': True, 'full_linear90': True, 'step_selection_within5': True}. Require all36 exact-oracle minus predicted-step upper losses <5pp; all9 unrestricted-linear recovery lower bounds >90%; and all36 true-step minus predicted-step upper losses <5pp. All are evaluated using one simultaneous81-contrast band. Unit-step results are descriptive and remain saved, including failures.

Document audit: {'excluded_unique_documents': 5824, 'new_unique_documents': 512, 'full_document_and_window_hashes_verified': True}.

Audits: {'payload_geometry_max': 4.944789111505088e-07, 'central_difference_linear_derivative_max_relative': 9.600462109302053e-07, 'central_difference_sites_per_condition': 4, 'central_difference_step': 0.0001}. The central-difference check uses the first four fixed sites of all39 conditions with double-precision arithmetic and explicitly the native float32 epsilon; it checks the derivative implementation, not approximation accuracy at unit intervention. Fresh finite responses and exact-response oracle grids are recomputed and saved. Input hashes verified; full16/64 ranks retained; maximum normal-equation residual 4.8e-12. Independent saved-array checks verify that predicted choices minimize predicted KL, errors use those choices, true-target oracle choices are kept separate, and baseline scores agree across linear and exact-response grids.

The predicted-target safeguard is not guaranteed to improve true-target KL over baseline. Cases where it worsens the true score are retained and counted in the independent audit, along with fixed-step failures and coefficient norms. Prediction arrays, true/predicted grids, chosen indices, unrestricted errors and bootstrap samples are saved. Independent audit fields are added by this reporting script.

Uncertainty: 4,000 paired recipient-document/control-draw bootstrap resamples; joint81 bands cover36 coefficient-path losses,36 step-choice losses,9 unrestricted-linear recoveries. Fixed bases and empirical donor assignments. Fresh documents from the same cached corpus, not a model-training holdout or cross-model/domain test. All three H389 gates and the simultaneous81-contrast procedure are unchanged. Figure shows rank64 point estimates and unrestricted predictions; no semantic-unit claim follows from approximate derivative sufficiency.

All three unchanged frozen gates PASS on the fresh panel. Unrestricted linear prediction recovers96.72–99.25% of native finite KL effect across the nine head/family groups; the smallest simultaneous lower bound is95.95%, above90%. Rank64 predicted-response coefficients/step selection recover70.01–84.46% versus exact-response oracle71.27–85.23%. Across both ranks and bases the practical-path loss is0.37–2.92 percentage points, with largest simultaneous upper3.70 points, below5. True-target step choice improves over predicted-target choice by0.04–1.95 points; largest simultaneous upper2.72 points, also below5.

Retained failure replicates: unit-step rank16 recovery falls as low as−341.45%, whereas predicted-target guarded rank16 recovery is48.57–71.48% and the exact-response oracle is49.69–72.30%. The practical step selector only sees the predicted response, not the actual finite target, but is essential for aggregate reliability at rank16. It still does not guarantee improvement at every individual site. All negative cases and individual violations are retained in the arrays and audit.

This independently confirms that a contextual first-order response generator, coupled to recipient output geometry and prediction-based step selection, captures most of the earlier oracle recovery for this intervention family. The predictive path requires intact context, the specified source payload and the full native weights. It does not show that the model is intrinsically rank64, that arbitrary coordinates are semantic features, or that this procedure is cheaper than native computation. The unrestricted derivative result and the low-rank projection result are distinct claims.

Twenty refreshed hypotheses and next discriminators:

|#|Hypothesis|Evidence and discriminator|
|---:|---|---|
|1|The contextual first-order derivative predicts most current MLP-only intervention behavior.|Supported on two panels; test larger intervention amplitudes.|
|2|The high low-rank recovery requires the exact finite target for coefficients.|Disfavored for this family by fresh confirmation; inspect more difficult interventions.|
|3|The high recovery requires exact-target step selection.|Disfavored at five-point aggregate accuracy here; retain per-site failures.|
|4|A fixed unit step is adequate at all ranks.|Contradicted by replicated rank16 aggregate failures.|
|5|Rank64 spans are uniquely meaningful semantic coordinates.|Weakened by random-basis success; test interpretation and causal specificity separately.|
|6|Bilinear self-interaction becomes essential at larger amplitudes.|Unresolved; independently toggle the quadratic response term.|
|7|Input RMS curvature is the first source of linear-model failure.|Unresolved; exact versus first-order input normalization at fixed amplitudes.|
|8|Final RMS curvature dominates failure.|Unresolved; exact versus first-order output normalization.|
|9|The three curvature sources compensate rather than add.|Plausible; use the full factorial, not a one-at-a-time attribution.|
|10|Finite softmax excursions, rather than response error, drive low-rank failures.|Supported by earlier diagnostics; compare unrestricted versus restricted predictions.|
|11|Prediction-based finite safeguards generalize across amplitudes.|Unresolved; freeze the rule before the amplitude ladder.|
|12|Only a handful of likely-token constraints are needed.|Disfavored under the existing solver; top32 losses remain.|
|13|Hundreds of context-selected token constraints approximately suffice.|Supported by H384 fresh top512 test; combine with the predictive path later.|
|14|Concentration alone determines projection success.|Disfavored by probability-to-token reassignment controls.|
|15|Tail anisotropy can be replaced by scalar stabilization.|Useful but insufficient at the declared five-point margin.|
|16|One population-average tail template removes context dependence.|Not established; tested mean templates fail universal criteria.|
|17|Native bilinear factor contributions are sparse for each local derivative.|Unresolved; measure context-dependent contribution sparsity after validating response prediction.|
|18|The same small factor set transfers across contexts.|Unresolved and stronger than per-context sparsity; separate selection and evaluation documents.|
|19|The final-layer result transfers to earlier layers with downstream propagation.|Unresolved; would require matched intervention/evaluation controls.|
|20|Context-conditioned linear operators are a useful computational unit, rather than fixed feature axes.|Supported as a predictive description in this regime, not a unique ontology; test boundaries and executable complexity.|

Next priority is the boundary of linear prediction, not another covariance refinement. Freeze an amplitude ladder and a full factorial of input-normalization curvature, bilinear quadratic response, and final-normalization curvature. Holding the intact direct residual fixed, define normalized input change u; the bilinear response is2B(x,u)+B(u,u). Independently choose exact/linearized input RMS, retain/remove B(u,u), and exact/linearized final RMS. The all-exact corner must replay the true finite intervention. Score unrestricted predicted responses first to isolate response-model failure from span projection and step selection. Preserve negative recoveries and distinguish algebraic decomposition from a learned compressed model. Fresh panels after this run must exclude6,336 complete-document hashes. Broad goal remains active.
