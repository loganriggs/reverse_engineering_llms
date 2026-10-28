## H442 — Baseline probability weighting gives only a small transfer gain

H441 showed that broader activity calibration improved activity transfer without recovering exact-map behavior. H442 keeps rank64, the384/128 calibration split, the50/50 empirical/isotropic input objective and exact radial correction fixed. It replaces the output Euclidean metric with an average baseline Fisher metric estimated from128 probability-sampled token gradients per calibration context. Token gradients are centered by the exact full-vocabulary expected gradient before sampling. The sketch is an unbiased conditional metric estimator, but finite-sample estimation error remains.

For each ridge multiplier1,.1,.01,.001 times mean metric eigenvalue, form a regularized metric and solve its generalized rank64 activity approximation using an encoder/decoder pair. Include the unweighted candidate. Select using reserved baseline predicted-map metric error, not finite target responses. Compiling encoder*T_H maps plus the decoder has the same1,253,376 coefficient count as the orthogonal rank64 scheme; an oblique projection does not require storing a second runtime encoder once maps are compiled. Baseline model, retained factors, transforms and calibration costs remain outside that count.

| Candidate | Training sampled-metric error fraction | Validation sampled-metric error fraction |
|---|---:|---:|
|unweighted|0.188153|0.186959|
|metric_ridge_1|0.134619|0.171412|
|metric_ridge_0.1|0.124635|0.181235|
|metric_ridge_0.01|0.125304|0.181663|
|metric_ridge_0.001|0.125402|0.181858|

Validation selects ridge multiplier1, the strongest regularization in the frozen grid. Weaker regularization does not improve reserved validation. An independent full-vocabulary metric audit, without reselection, finds validation error fractions18.54% unweighted and16.99% selected, close to sampled18.70%/17.14%. Thus the selected calibration improvement is present in the full metric too; it is not merely an artifact of sampled validation scoring. This audit does not rule out noise in the training metric estimate or establish population significance.

| Frozen finite-KL contrast | Estimate | Joint95% interval | Gate passes |
|---|---:|---|---|
|selected_minus_unweighted_low|+0.000972|[-0.066064, +0.068007]|False|
|selected_minus_unweighted_median|+0.003255|[-0.063780, +0.070291]|False|
|selected_minus_exact_low|-0.049401|[-0.116436, +0.017635]|False|
|selected_minus_exact_median|-0.022505|[-0.089541, +0.044530]|False|

All joint gates fail: neither >2-point transfer improvement over unweighted compression nor within2-point noninferiority to exact maps is established in either stratum. Finite-KL recovery changes from93.17% to93.27% low and96.83% to97.16% median, versus exact98.21%/99.41%. The small positive point gains are not confirmed by these intervals. The bootstrap conditions on the selected projection, frozen calibration and donor pool; it resamples paired transfer sources/documents and does not resample model selection.

Unweighted scores replay H441 within1.04e-17 absolute difference. Encoder/decoder composition is idempotent to numerical precision in all candidates. Reserved calibration split and source/map hashes are inherited and checked; full-vocabulary audit uses the frozen selected projection. No new documents, no prior gate changes, total12480.

Interpretation: making a fixed global compression probability-sensitive improves its baseline calibration objective, but the observed finite transfer gain is small. This does not prove a fixed rank64 output span inadequate: averaging output metrics across contexts discards their association with particular baseline activities, and a fixed projection rule may be the limiting choice. Conversely, good baseline metric fits do not by themselves identify meaningful semantic units.

Next distinguish span capacity from projection-rule failure. Keep a fixed rank64 output span, but compute its projection coefficients using each receiving context’s baseline Fisher metric and analytic linear-map responses. Compare the original fixed projection, the context-conditioned projection, and an equally sized fixed random output span. No finite target errors enter coefficient choice; finite KL remains the evaluation endpoint. Context-conditioned projection is an expensive diagnostic, not a compiled speedup. If it helps even random spans, that would implicate output-metric geometry rather than privileged feature directions; if learned spans help uniquely, their structure merits further study. An efficient predictor would still need to reproduce that behavior without a dense context-specific solve.
