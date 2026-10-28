## H436 — Input-derived prediction of omitted directions

H435 showed that the omitted union128 response is small in raw magnitude but disproportionately important under output scoring, and scalar correction cannot recover it. H436 tests first- and second-order Taylor predictions of that omitted response, using only baseline pre-MLP state, source vector and model weights. Add the predicted omitted response to the exact native128-factor response, then evaluate against the full finite response. Full target responses are used for scoring, not for fitting or choosing predictor coefficients.

The first-order predictor uses the native-epsilon RMS derivative and bilinear product rule. The second-order predictor includes both the quadratic bilinear term and the second-order normalized-input coefficient. Omitted factor indices remain the fixed complement of union128. Wrong-source controls cycle the linear prediction within each six-source family while keeping recipient context fixed. All32 H434 origins and destinations are evaluated; frozen primary statistics concern matched contexts. No new documents; total11968.

| Matched omitted Fisher recovery | Estimate | Joint95% interval | Lower >90% |
|---|---:|---|---|
|linear_low|96.988%|[91.289%, 102.686%]|True|
|linear_median|99.548%|[93.849%, 105.247%]|True|
|second_order_low|99.755%|[94.056%, 105.454%]|True|
|second_order_median|99.993%|[94.294%, 105.692%]|True|

All four frozen >90% gates pass. Bands use4000 paired source/document resamples conditional on the baseline-selected strata and frozen donor pool. Symmetric maxabs bands extend above the physically attainable recovery ceiling1; those upper endpoints do not imply recovery above100%. This is a new predictive test on reused data, not another fresh-document confirmation.

| Matched stratum | Correction | Fraction of previous finite-KL error removed | Full finite-KL recovery |
|---|---|---:|---:|
|low|Linear|96.578%|98.669%|
|low|Second order|99.689%|99.879%|
|low|Wrong-source linear|-55.561%|39.482%|
|median|Linear|99.540%|99.929%|
|median|Second order|99.993%|99.999%|
|median|Wrong-source linear|-70.224%|73.903%|

For low-RMS matched contexts, full finite-KL recovery rises from61.10% for union128 alone to98.67% with the linear omitted correction and99.88% with second order. Median contexts rise from84.67% to99.93% and99.999%, respectively. Wrong-source corrections reduce coverage below union128 alone, supporting source specificity of this predictor rather than a generic beneficial vector. Wrong-source results are descriptive controls outside the frozen joint4 coverage family.

An independent central-difference audit checks the second-order RMS coefficient. Pooled relative discrepancy decreases approximately fourfold when the step halves from0.04 to0.02 to0.01:5.66e-5,1.41e-5,3.54e-6, as expected for this check. At step0.001 pooled discrepancy is3.78e-8, while maximum per-site relative discrepancy is2.89e-4 and worsens at smaller steps because tiny coefficients become subtraction-roundoff sensitive. No sites are discarded; this diagnostic is not a newly invented pass gate.

**Cost limitation:** the predictor computes learned reads for all4608 native factors and projects all4480 omitted factors. The second-order version requires additional reads. This is a dense analytic account, not an efficient128-factor computation, global model compression, or a discovered semantic feature set. The exact retained response still requires its own finite coefficient changes. High predictive fidelity establishes that the missing behavior is accessible from baseline state and incoming source, but not that it has a small transferable representation.

Next: form the omitted local input-output operator from baseline state and weights, restricted to the known head6 source subspace. Test whether a small basis chosen from that operator and baseline output geometry predicts unseen source responses, compared with a shared basis and size-matched controls. Crucially, choose directions without full finite target errors or per-example oracle fitting. Account separately for basis-construction cost, coefficient evaluation, storage and transfer across contexts. A context-local operator representation may be compact to describe yet expensive to construct; that distinction is part of the research question.
