## H875 — Final-state normalization explains much of the format-dependent gain failure

H874 left the failed frozen gains unchanged and proposed a parameter-free normalization explanation. H875 captures native final residual states and tests it. The mean copying-to-capital RMS ratio changes from0.879 in familiarA/B formats to1.165 in the new competentC format. Substituting the per-context RMS ratio for the old fitted gain reduces late-reader newC response-transfer errors from27.1–29.8% to4.5–10.0%. This supports normalization as a major source of the format dependence, without establishing it as the complete mechanism.

### Models being compared

The target remains the symmetric finite-amplitude response to the same head-write direction in matched copying and capital contexts. For each endpoint, the simple parameter-free prediction is:

    predicted capital-context response
      = measured copying-context response × RMS(copy final state) / RMS(capital final state).

The RMS values come from unmodified native forwards, not task-output fitting. Both contexts use the same actual source direction and common perturbation amplitude. Ratios are applied before canonical orientation and averaging within lexicalpairs. Like earlier transfer tests, this predictor consumes a measured copying intervention response; it is not an intervention-free prediction of the complete model.

A second, more detailed control freezes all attention/MLP updates after the injected head write, including the current layer's MLP. It propagates the write only through the residual skip multipliers, then applies the actual final RMS normalization, unembedding and30tanh logit cap. Its predicted final-state change is product(later residual multipliers) × W_O_head × head_edit. We compare its analytic response ratio between contexts with live responses, and separately evaluate its absolute response prediction. Fixed-native-RMS and no-logit-cap variants are retained to identify what the approximation includes. No variants are fitted.

### New competent formatC: relative response-transfer RMSE

|Reader|Output contrast|Old frozen gain|Native RMS ratio|Exact frozen-update response ratio|
|---|---|---:|---:|---:|
|L13H0|country|0.276|0.063|0.092|
|L13H0|capital|0.298|0.100|0.159|
|L14H4|country|0.271|0.045|0.082|
|L14H4|capital|0.272|0.094|0.065|
|L8H3|country|0.289|0.272|0.297|
|L8H3|capital|0.395|0.271|0.303|

For the late readers, RMS ratio gives4.5–10.0% error versus27.1–29.8% with frozen gains. On new pairs in familiarA/B, RMS errors are7.4–15.0%, broadly comparable but not uniformly superior to the previously fitted7.5–9.4%. L8H3 remains poorly captured inC at27.1–27.2%, so normalization is not a general replacement for downstream computation.

The mean native RMS ratios are0.879 for both old and recombinedA/B contexts,1.165 for newC, and0.978 for failed-capitalD. Equality of the old/recombinedA/B aggregate is expected because recombination reuses the same individual country prompts; the paired contrast changes, but the single-country native contexts are largely reused. The formatC shift, rather than lexicalpair recombination, is the novel context-scale change.

The more detailed frozen-update ratio is not uniformly better than simple RMS ratio. InC it gives6.5–15.9% error for late readers and29.7–30.3% forL8H3. Exact readout algebra can still be an inaccurate model of live responses if later computation changes. Including a known operation is not automatically beneficial when other omitted effects compensate for it.

### Absolute prediction and limits

Against live endpoint responses inC, the exact frozen-update model has6.0–19.1% relativeRMSE for late-reader task/output combinations and16.0–38.0% forL8H3. These absolute errors are distinct from ratio-transfer errors: a ratio can work by cancellation even if both underlying response predictions are imperfect. The remaining differences may reflect later MLP/attention responses, radial normalization effects, logit saturation and finite-amplitude nonlinearity; their individual contributions are not yet identified.

All denominator cases are retained: no analytic-ratio denominator is below1e-10, and the minimum absolute skip-response denominator is0.748. Thus the reported improvements are not produced by dropping unstable ratios. FailedcapitalD remains a separate diagnostic; competentA/B/C results are not inferred from it.

### Research interpretation and next step

The compact description now has three distinct parts: a static content direction, a signed routing coefficient, and a context-dependent normalization scale. This is relevant to a data-based functional representation: the same raw write can have a different effect solely because the surrounding residual state changes its effective scale. It does not imply that all remaining dimensions are causally irrelevant or that a unique canonical representation has been found.

H875 is a post-failure mechanistic diagnostic on already observed contexts. No parameters were fitted, but the normalization hypothesis was chosen after the failed confirmation. To test its causal role directly, retain the live downstream computation while clamping the final normalization denominator to a shared value in matched copying/capital contexts. Compare equal-write responses with native normalization versus this clamp, including the actual readout and logit cap. Do not interpret the clamp as a naturally occurring model state. If substantial differences remain, localize later computational changes rather than adding another fitted format gain.

The task-blind discovery of reusable features/manifolds and broad-task generalization remain unresolved. These results guide the metric and normalization information that a useful featurizer should preserve; they do not validate a learned group ontology.

### Verification

336 new native whole-model forwards, reusing existing live-intervention responses. Native replay exact; selected-token exact readout error6.54e-06; iterative versus product residual-skip propagation error1.14e-13. All source hashes unchanged, outputs finite, no reserved-test use;26,112-document pool unchanged. Parameter-free ratio analyses and absolute errors saved with4096 uncorrected lexicalpair-bootstrap intervals where applicable. Readers, tokens and panels remain scoped as before; no claim of independent global research holdout. Broad goal remains active.
