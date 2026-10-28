

## H389 — Input-derived first-order response prediction

Predict the derivative at t=0 of RMS(pre + B(RMS(pre−t·payload), RMS(pre−t·payload)) + Down_bias). This holds the direct residual pre fixed and matches the existing MLP-only source-removal target. Differentiate input RMS, the bilinear map, and final RMS analytically, using the original float32 RMS epsilon. Prediction uses intact residual, source payload and native weights; it never reads the exact finite response. Source controls, heads, four positions and512 documents are the same H384 panel.

Fit ranks16/64 in the frozen probability/random bases to the predicted normalized response using the recipient probability metric without cap derivative. Compare four paths: cached exact-response coefficient/step oracle; linear coefficients with true-target step oracle; linear coefficients with step selected solely against the predicted full response; and linear coefficients at unit step. The predicted-target step uses the same eight-step grid and full capped-vocabulary scoring, with the predicted distribution as its reference. Also score the unrestricted linear prediction. Exact finite targets are used only for evaluation and the explicitly labeled oracle controls. No second RMS is applied after the predicted normalized displacement. This removes exact finite target access from the practical path, but retains native weights and the known intervention payload: it is not parameter compression or an unconditional next-token predictor.

|Basis|Head|Family|Exact oracle64|Linear true-step64|Linear predicted-step64|Linear unit-step64|Full linear|
|---|---|---|---:|---:|---:|---:|---:|
|probability|0|actual|0.8599|0.8536|0.8530|0.8529|0.9927|
|probability|0|moment|0.7478|0.7376|0.7362|0.7362|0.9842|
|probability|0|donor|0.7479|0.7376|0.7363|0.7363|0.9842|
|probability|1|actual|0.8012|0.7966|0.7962|0.7962|0.9892|
|probability|1|moment|0.7901|0.7843|0.7835|0.7835|0.9860|
|probability|1|donor|0.7919|0.7863|0.7856|0.7856|0.9868|
|probability|6|actual|0.8158|0.8027|0.7932|0.7888|0.9759|
|probability|6|moment|0.8352|0.8321|0.8194|0.8222|0.9733|
|probability|6|donor|0.8396|0.8372|0.8230|0.8277|0.9733|
|random|0|actual|0.8132|0.8061|0.8050|0.8049|0.9927|
|random|0|moment|0.7287|0.7191|0.7173|0.7174|0.9842|
|random|0|donor|0.7272|0.7172|0.7156|0.7157|0.9842|
|random|1|actual|0.7737|0.7701|0.7696|0.7696|0.9892|
|random|1|moment|0.7610|0.7567|0.7558|0.7559|0.9860|
|random|1|donor|0.7638|0.7597|0.7589|0.7590|0.9868|
|random|6|actual|0.7738|0.7627|0.7519|0.7499|0.9759|
|random|6|moment|0.8056|0.8037|0.7896|0.7961|0.9733|
|random|6|donor|0.8128|0.8116|0.7967|0.8040|0.9733|

Frozen gates: {'predicted_fit_within5': True, 'full_linear90': True, 'step_selection_within5': True}. Require all36 exact-oracle minus predicted-step upper losses <5pp; all9 unrestricted-linear recovery lower bounds >90%; and all36 true-step minus predicted-step upper losses <5pp. All are evaluated using one simultaneous81-contrast band. Unit-step results are descriptive and remain saved, including failures.

Audits: {'payload_geometry_max': 4.959132411386236e-07, 'native_response_replay_max_absolute': 0.0, 'central_difference_linear_derivative_max_relative': 8.936338265037773e-07, 'central_difference_sites_per_condition': 4, 'central_difference_step': 0.0001}. The central-difference check uses the first four fixed sites of all39 conditions with double-precision arithmetic and explicitly the native float32 epsilon; it checks the derivative implementation, not approximation accuracy at unit intervention. Cached finite responses replay from regenerated payloads. Input hashes verified; full16/64 ranks retained; maximum normal-equation residual 1.7e-12. Independent saved-array checks verify that predicted choices minimize predicted KL, errors use those choices, true-target oracle choices are kept separate, and baseline scores replay H384.

The predicted-target safeguard is not guaranteed to improve true-target KL over baseline. Cases where it worsens the true score are retained and counted in the independent audit, along with fixed-step failures and coefficient norms. Prediction arrays, true/predicted grids, chosen indices, unrestricted errors and bootstrap samples are saved. Independent audit fields are added by this reporting script.

Uncertainty: 4,000 paired recipient-document/control-draw bootstrap resamples; joint81 bands cover36 coefficient-path losses,36 step-choice losses,9 unrestricted-linear recoveries. Fixed bases and empirical donor assignments, reused panel, no fresh confirmation. Figure shows rank64 point estimates and unrestricted predictions; no semantic-unit claim follows from approximate derivative sufficiency.

All three frozen gates PASS. The unrestricted first-order prediction recovers97.33–99.27% of native finite KL effect across the nine head/family groups, with minimum simultaneous lower bound96.78%, above the90% requirement. This shows that for these source-removal magnitudes, the contextual local derivative closely predicts the full MLP-only behavioral response, including normalization effects.

Using the predicted response both for coefficients and for finite-step selection recovers71.56–85.30% at rank64 versus exact-response oracle72.72–85.99%. Across both ranks and bases the point loss is0.36–2.26 percentage points; the largest simultaneous upper loss is2.81 points, below5. The true-target step oracle improves over predicted-target selection by0.04–1.59 points; largest simultaneous upper bound2.14 points. Thus access to the exact finite response is not necessary for most of the observed low-rank recovery in this panel. The prediction still needs the intact context, intervention payload and native weights.

IMPORTANT retained failure: unit-step rank16 recovery ranges from−312.26% to72.10%. In contrast, predicted-target guarded rank16 recovery ranges50.48–72.15%, compared with exact-response oracle51.51–72.92%. Rank64 unit-step aggregate recovery remains71.57–85.29%, but this does not justify removing the guard generally. The low-rank local fit can extrapolate badly through finite softmax even when the unrestricted response prediction is accurate. A safeguard chosen against the PREDICTED response, without the exact target, is sufficient for the aggregate five-point criterion here. It is not a guarantee of per-site nonworsening against the real target; those violations remain in the saved audits.

The central-difference audit agrees within8.94e-7 relative error at four fixed sites per condition, and regenerated finite responses replay exactly. This supports the derivative implementation; the full-panel native KL results establish approximation quality separately. All Gram directions retained and normal-equation residuals below1.7e-12. The observed success cannot be attributed to exact finite targets leaking into the practical coefficient/step choices: their data path uses only the linear prediction; the true-grid arrays are evaluated afterward and used solely for scoring and labeled oracle controls.

Hypothesis update: the full-response oracle objection to earlier recovery results is substantially reduced for these interventions. A contextual first-order map supplies a good response, while recipient output geometry selects which errors matter; moderate-dimensional arbitrary spans can approximate that predicted response. Fixed semantic axes are still not identified, and the predictor uses the complete native bilinear weights. The derivative factorization D[(Lx)⊙(R dx)+(R x)⊙(L dx)] supplies an explicit context-conditioned computation, but calling its individual factors semantic features requires separate evidence. This is not a claim of weight compression, cheaper execution, or general success at larger perturbation amplitudes.

Next: freeze the predictor, metric, bases, and predicted-target step rule, then independently confirm on a fresh512-document panel excluding all5,824 used complete-document hashes. Keep all three gates, both ranks and bases, and explicitly retain unit-step failures. After confirmation, use an intervention-amplitude ladder to locate where linear prediction fails and decompose that failure into input-normalization curvature, bilinear quadratic terms, and final-normalization curvature. Those are more discriminating next tests of computational structure than further small covariance-template refinements. Broad goal remains active.
