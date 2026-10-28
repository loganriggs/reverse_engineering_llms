## H473 — Fresh receiving-document confirmation of compiled canonical maps

Freeze H472's validation-selected64/32 and128/32 maps, represented by their compiled768x16 kernel reads and16x1152 writes. No fitting or selection on this panel. Evaluate512 new complete-document-hash-disjoint receiving windows at token position63, with the same64 centered head6 donor vectors assigned eight times each and both signed unit interventions. Sources, model and calibration are unchanged. Labels for CE are the observed next token. Both intact-state reconstruction and delta prediction around the native intact baseline are measured.

The original20k-row token cache had only31 eligible unused complete documents left, so the512-document builder stopped before saving any panel. A separate10k-row FineWeb cache was built from stream documents65000–72935, using the same GPT-2 tokenizer snapshot.512 complete documents were sampled with seed20271128. An independent audit rehashed all12992 excluded documents and all512 new ones against their respective original token streams, verified every129-token window, and found zero overlap. Total tracked completed documents is now13504. This is fresh receiving-document confirmation for the tracked continuation, not a model-training holdout or new source-family test.

| Frozen map | Negative KL / raw-response recovery | Positive KL / raw-response recovery | Intact KL | Intact CE change | Top1 agreement |
|---|---|---|---:|---:|---:|
|r64|80.84% / 92.48%|76.73% / 92.26%|0.144187|+0.121961|83.984%|
|r128|81.29% / 93.35%|77.93% / 93.27%|0.123766|+0.103293|85.352%|

| Frozen primary statistic | Estimate | Joint95% interval | Pass |
|---|---:|---|---|
|r64_absolute_negative|+0.808367|[+0.728495, +0.888239]|False|
|r64_absolute_positive|+0.767335|[+0.716151, +0.818518]|False|
|r128_absolute_negative|+0.812920|[+0.729304, +0.896536]|False|
|r128_absolute_positive|+0.779261|[+0.730010, +0.828512]|False|
|width_gain_negative|+0.004553|[-0.010677, +0.019783]|False|
|width_gain_positive|+0.011926|[-0.000548, +0.024400]|False|
|intactKL_wider_minus_narrow|-0.020421|[-0.027961, -0.012881]|True|

The first four gates require absolute probability-response recovery lower>.8; both widths fail at both signs. Wider-minus-narrow response gains must exceed5pp; gains.455/1.193pp have joint intervals[−1.068,1.978]/[−.055,2.440]pp and fail. The intact-KL increase threshold upper<.01 passes: wider-minus-narrow change−.020421, interval[−.027961,−.012881]. Thus wider coordinates improve intact distributional fidelity on this panel, but no response advantage is established.4000 paired bootstrap samples resample eight receivers within each of64 fixed donor strata, preserving the balanced donor design. Joint7 standardized bands condition on the frozen donor pool and all prior fitting/selection.

Response point estimates77–81% are higher than the earlier reused-panel~62–64%, but the80% lower-bound criterion remains unmet. This is evidence of meaningful transfer, not uniformly high response faithfulness. A change in point estimate across panels is not itself a causal explanation or proof of distribution shift. Mean effect KL is.00261197 for negative and.00198807 for positive amplitude; raw-response recovery remains~92–93%.

### Concentration and sensitivity

Negative effects are highly concentrated: the largest case contributes20.94% of total effect KL and the top26 cases65.46%; the squared-weight effective case count is16.58 out of512. For positive effects these quantities are12.07%,55.35%, and36.19. Effective counts are descriptive concentration measures, not replacements for the bootstrap sample size. Error KL is also concentrated, with effective counts~36–49 across methods/signs.

Removing any one receiver leaves wider-model negative recovery between78.62% and83.00%, and positive between77.10% and78.81%. The increase relative to the old panel therefore cannot be attributed entirely to a single new outlier. These are post-evaluation descriptive diagnostics and do not change the primary scores, intervals or gates. Aggregate recovery emphasizes effect-heavy events and should not be interpreted as the fraction of contexts explained equally well.

The compiled maps agree with their unfused float32 reference predictions on the new intact/intervened inputs within the frozen2e−6 relative tolerance. This extends the H472 algebra check to new data and finite inputs. The reference dictionary is still part of each model; a16-coordinate final read/write factorization is not a16-neuron model. Exact native baseline states remain only in the response-isolation metric; full-state predictions use learned writes directly.

Execution note: the new cache builder printed its completed manifest but remained live after saving. Cache bytes were independently verified against that manifest and all document windows before use. The verified builder process was then terminated during shutdown; its exit143 is recorded separately from successful corpus integrity checks. No GPU experiment was restarted. Initial cache-exhaustion attempts saved no panel. Old cache files were unchanged.

Next prioritize explaining receiver heterogeneity before further tuning. On saved states, measure how much each intervention's degree-two input-feature difference lies outside the frozen empirical input subspace, together with source magnitude, source/residual angle and baseline probability sensitivity. Compare these geometry-only predictors with errors and effect concentration. Use the earlier panel for any learned error predictor and label H473 analysis as post-hoc cross-panel diagnosis, not a newly pristine test. This can distinguish missing input support from downstream sensitivity or a few difficult receiving regimes. Future confirmation must use further new documents. The original CE hierarchy, source rank1/tail question and broader-head/model transfer remain open.
