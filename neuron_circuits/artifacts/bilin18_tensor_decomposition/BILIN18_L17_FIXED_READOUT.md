

## H280 — Existing read coordinates help modestly; readout estimation remains a limitation

H279 showed diminishing tested gains from constant-only optimization. H280 adds a state-dependent linear correction using ONLY the existing natural core32 and actual-trained partner32 read coordinates. The original compiler, H279 refined constant, mean, core/partner bases and quadratic terms remain fixed. Features are centered and standardized using the first768 discovery documents. Correction is f(x)W, W64x1152, with no fitted intercept. Centering makes the mean added write zero on discovery states; it does not leave every individual state or intervention response unchanged.

Five output-MSE ridge fits (0/.01/.1/1/10) and eight KL runs are fixed in advance. KL runs cross zero/MSE initialization, rates.3/3 and correction-energy penalty0/.1,400 updates, batch32, checkpoints0/100/200/300/400. The penalty is mean squared added-write norm divided by training baseline-error energy. Both fit families select by the same next256-document validation KL, so the comparison does not use different selection metrics. MSE selects ridge1; KL selects MSE initialization/rate.3/penalty.1 at step300. All candidates retained, including step0.

Fresh128 documents exclude2368 prior complete-document hashes; uniqueness, exclusions and file SHA256 verified. Not a training or global holdout. A fixed one-document cyclic shift of KL-adapter features preserves each position slot and the feature marginal distribution while mismatching state context. This is a diagnostic perturbation, not a deployable model, semantic intervention or proof of a particular causal feature. Original base writes stay matched to their own inputs.

| Model | Training KL | Validation KL | Fresh KL | Fresh CE increase | Native top1 agreement |
|---|---:|---:|---:|---:|---:|
|constant|0.048954607920084416|0.056146855700013446|0.056580|0.04231|86.91%|
|mse|0.045833346928299365|0.05346074185183678|0.054300|0.04259|86.33%|
|kl|0.034372810313234047|0.05203628660366206|0.053294|0.03057|86.72%|
|shuffled|—|—|0.062846|0.04225|85.94%|

Locked gates: state_gain: FAIL, objective_gain: FAIL, state_specificity: FAIL, repair: FAIL.

- KL adapter versus constant: 5.81% relative KL reduction, paired document-bootstrap95% [2.97,8.57]%.
- KL adapter versus mse: 1.85% relative KL reduction, paired document-bootstrap95% [-0.95,4.69]%.
- KL adapter versus shuffled: 15.20% relative KL reduction, paired document-bootstrap95% [11.72,18.44]%.

The5.81% gain versus constant-only is positive but below the20% state-gain threshold. The1.85% gain versus MSE has an interval including zero and misses10%. The15.20% gain versus shuffled context is positive but below20%. Repair fails: total KL recovery versus the original compiler45.39% is below80%, CE+.0306 nats exceeds.02 and top1 agreement86.72% is below98%. No gates relaxed. The shuffled comparison establishes dependence on matched input context in this diagnostic, not semantic identification or transfer to a different population.

**Estimation versus feature limits.** KL fitting reduces training KL from.04895 to.03437 (about30%), but validation only.05615 to.05204 and fresh test.05658 to.05329. The larger fitted-readout gap makes estimation/regularization a live explanation; weak held-out gains do not by themselves show the64 coordinates lack useful information. Selected MSE and KL models have similar held-out behavior despite different training improvements. Rate3 runs deteriorate sharply, which is retained in the curves rather than omitted. This is not a global optimization or class-capacity bound.

**Costs and algebra.** W adds73,728 coefficients and128 normalization scalars, no new input directions or quadratic products. Total correction+compiler storage becomes3,769,120 scalars if kept as a separate adapter. Algebraically it can be folded into the existing dense linear map and constant; this preserves function and storage shape but changes their coefficients. The distinct adapter is retained for attribution. It changes source responses by a linear term, so previous response-preservation claims for constant-only calibration do not apply. Source-response transfer would need remeasurement before promoting this model.

**Next discriminator.** Hold these64 features and geometry fixed and run a discovery-label learning curve (e.g.192/384/768 documents) with the same validation panel, objective grid and explicit update budget. Geometry already uses768 documents, so this is a supervised-label-count axis, not total data exposure. Track train/validation loss and selected regularization. If held-out gains grow materially with labels while training gaps shrink, prioritize more estimation data; if they saturate under well-behaved fitting, compare new nonlinear features or input directions. Use a fresh test only after choices are frozen, and recheck intervention responses for any promoted state-dependent adapter. No semantic claims follow from successful regression alone.

**Audits.** Three focused tests pass: adapter response is exactly linear and uses only the stored read span, ridge satisfies regularized normal equations, and native-to-prediction KL gradients. Maximum reported ridge normal-equation residual3.8e-9 (audit arithmetic includes float32 identity scaling; fits use float64). Original program/bases/constants unchanged. Five ridge fits and eight KL runs complete; coefficients, normalizers, curves, selections and raw test errors saved. Full project suite not rerun, figure inspected. Broad research goal remains active.
