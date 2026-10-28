## H484 — Direct finite-KL refinement at fixed representation

Freeze H482's PCA20 output basis and256 input coordinates. Start from its validation-selected Fisher coefficient fit and optimize actual finite target||prediction KL on the same1024 training receivers and both signs. No new predictor parameters or directions. Use16 full-batch gradient steps, normalizing the coefficient gradient so unit step changes the training predicted-response norm by its current norm. Training line search fractions [.001,.01,.03,.1] include no update. Select the best stopping point on the same256 validation receivers, including step0. Save maps and selection before collecting outcomes on512 newly reserved receivers.

This refinement removes the explicit ridge penalty and uses initialization plus early stopping as regularization. Therefore it tests this finite-KL refinement procedure, not a perfectly isolated objective comparison under identical regularization. The differentiable objective includes native final RMS normalization, vocabulary projection, softcap and softmax. There is no convergence guarantee and16 steps cannot establish the best attainable finite-KL fit.

Independent complete-document audit verifies16320 excluded documents and512 new documents against their respective token caches, checks129-token windows and finds zero overlap. Total tracked documents16832. Fixed training donors32, evaluation donors64, source calibration and intervention signs unchanged. Freshness is within this continuation, not a model-training holdout.

Validation selects step1. Subsequent steps keep improving training loss, but all have worse validation than step1. The early selected step is evidence against a large transferable correction from this particular trajectory, not proof that local-Fisher and finite-KL optima coincide.

| Predictor | Fresh confirmation KL recovery −/+ | Raw recovery −/+ |
|---|---|---|
|initial|76.9533% / 73.4982%|93.7917% / 93.6241%|
|selected|76.8763% / 73.5017%|93.7879% / 93.6198%|

| Predeclared primary comparison | Gain | Joint95% interval | Lower>0 |
|---|---:|---|---|
|selected_minus_initial_negative|-0.0770pp|[-0.2119, +0.0579]pp|False|
|selected_minus_initial_positive|+0.0035pp|[-0.0373, +0.0443]pp|False|

Initial training/validation KL: 0.0002294890/0.0004379411. Selected step1: 0.0002292468/0.0004379127. Last step16: 0.0002269204/0.0004394453.

Both primary improvement gates fail. Confirmation changes are−.0770pp and+.0035pp, with joint intervals[−.2119,.0579]pp and[−.0373,.0443]pp. This procedure offers no established gain and leaves the previous learned-versus-oracle gap essentially untouched. These narrow paired-change intervals concern only this selected small update, not the uncertainty in absolute recovery or the attainable performance of other procedures. Recovery is pooled effect-normalized error reduction, not accuracy.

Bootstrap uses4000 paired samples, eight receivers resampled within each of64 fixed donor strata, with joint2 standardized bands shared across signs. It conditions on fitting, calibration, donors and validation selection. Effects remain concentrated: negative effective count13.26 with largest-case share25.50%; positive effective count27.01 with largest14.66%. Those are descriptive concentrations, not alternative sample sizes.

Audits: initial validation finite KL independently replays H482 within1e−7; all logged training losses are nonincreasing; all output arrays finite; document exclusion/window audit passed; selected maps saved before confirmation collection. Initial/selected/final coefficient maps, full line-search/stopping history and per-case outcomes archived. Native intact state remains the response-isolation baseline; no intact CE or full-layer replacement claim. No process remains active.

Hypothesis update: useful output capacity exists (H483), but adding it did not produce learned transfer (H482), and this direct finite-KL refinement did not repair that (H484). The current evidence gives limited support to objective mismatch alone as the dominant practical bottleneck under these methods. It leaves training coverage, regularization, input information and receiver-dependent coordinate targets unresolved. It does not justify calling the input representation inadequate or the optimizer converged.

Next take a broader diagnostic step before more width or loss tuning: refresh the competing hypotheses against the full canonical-form evidence, then test the long-standing source rank1/tail question. The centered natural source pool has~97.6% of its energy in one direction, but that is not a behavioral result. Decompose the same donor vectors into their fixed calibration top direction and orthogonal tail; intervene on each and their sum on already collected receivers, preserving the native nonlinear MLP. Measure their probability effects, raw effects and interaction. This can reveal whether small source components drive the difficult response remainder, or whether the dominant source coordinate already generates it through receiver-dependent computation. Freeze decomposition from existing calibration data, label the receiver analysis exploratory, and reserve fresh confirmation for any resulting learned claim. Keep canonical CE comparisons, coefficient learning and broader-head/layer transfer open rather than replacing the broad goal with this subexperiment.
