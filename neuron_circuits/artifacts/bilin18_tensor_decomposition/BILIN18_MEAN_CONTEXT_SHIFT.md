## H641 — A shared MLP-context shift predicts held-out operator effects

H640 was progress: the late-context/attention pattern replicated on new items and one-token, equal-length instruction contrasts. H641 replaces access to each exact donor MLP context with a simple fitted prediction.

For each family (copy/fact), template pair (matched/broad) and layer10–17, fit the mean change in normalized native MLP input over items0–3 only. Predict the donor operating context for any receiver as its own normalized context plus this mean shift. Reverse template exchange uses the exact negative shift. Items4–7 are excluded from fitting. This is32 discovery prompts and32 held-out prompts, because each item appears in four templates. The split and mean rule were fixed before this fit, but these items' H640 outputs had already been observed: this is retrospective item holdout, not a fresh blind replication.

The predictor does not fit output logits, response derivatives or any probe-specific parameter. It uses no held-out donor MLP context in its predicted path. The native receiver state, model weights, MLP normalization and attention are retained. Donor incoming tangents or readouts remain available in the explicitly donor-conditioned variants; receiver-conditioned variants use receiver components throughout except the predicted context shift. Test the same three probes as before. Exact donor-context replacement is the target intervention; no shift is the baseline. A control imports the mean shift from the other family at the same template index, rescaled to the fitted shift's norm.

The predicted normalized context is not renormalized: it is an additive approximation to a donor normalized vector. It need not lie exactly on the normalization sphere. This defines a local bilinear-operator approximation rather than a guaranteed naturally occurring model state.

**The mean shift predicts much of the held-out context-replacement effect.** The primary error is norm(predicted logit derivative − exact context-replacement derivative) divided by norm(exact context-replacement derivative − no-shift derivative), using receiver-probability-weighted centered logits. No shift therefore has error1. With donor incoming tangent/readout, prediction error is0.185–0.359 across the four cells, and all96 held-out prompt–probe cases improve over no shift. Receiver incoming/readout also improves in all96 cases, with mean errors0.186–0.395. These repeated observations are not independent trials.

| Held-out condition | Predicted effect error, donor input/readout | Other-family control error | Predicted effect error, receiver input/readout | Other-family control error |
|---|---:|---:|---:|---:|
|copy / matched|0.359|1.695|0.353|1.708|
|copy / broad|0.356|1.208|0.395|1.192|
|fact / matched|0.308|1.241|0.298|1.224|
|fact / broad|0.185|1.102|0.186|1.193|

The other-family shifts have the same norm but are worse than no shift on average in every cell and both endpoint conditions. This rules out shift magnitude alone as an explanation for the successful predictor. It does not establish semantic uniqueness: copying and factual prompts use different instruction words and formats, so the control changes several contextual attributes.

Results also hold across probe types. Mean held-out effect error under donor endpoints is0.315 for the historically selected original vector,0.285 for the learned-span probe and0.306 for the orthogonal probe. The mean model was fitted from native contexts without using any of these directional responses, although all three probe directions were previously studied. Generalization to new directions remains an explicit next check.

**Predicting the context effect is different from predicting the entire donor computation.** Under donor incoming tangent/readout, remaining full donor–receiver gaps are:

| Held-out condition | No context change | Exact context change | Predicted context change |
|---|---:|---:|---:|
|copy / matched|0.616|0.342|0.415|
|copy / broad|0.690|0.336|0.416|
|fact / matched|0.749|0.342|0.436|
|fact / broad|0.635|0.204|0.228|

The predicted intervention generally preserves much of the exact intervention's benefit, but both leave other factors unmatched. With receiver incoming tangent/readout, predicted remaining gaps are0.774,0.965,0.827,0.853 for copy-matched, copy-broad, fact-matched, fact-broad respectively. Successful prediction of a component effect is therefore not successful end-to-end reconstruction of the donor response. In fact-matched receiver endpoints, the mean approximation is slightly closer to the full donor than the exact context intervention; this is not superior component fidelity, since unrelated residual errors can cancel.

The underlying context shifts themselves transfer approximately: held-out mean Euclidean shift errors are0.338/0.378 for matched/broad copying and0.358/0.238 for matched/broad facts, with mean cosine similarities0.946/0.927 and0.931/0.968. Functional validation is more informative than these geometric similarities alone, and is reported separately above.

What has been represented is a template-pair-dependent *change in a local operator*. For a bilinear MLP, changing context by mu changes its derivative by B(mu,DN_x q). Thus a shared context shift defines a reusable operator-change rule for different input directions, conditioned on receiver normalization. Across layers these local rules compose; the full output effect need not be additive in the layer shifts. The model stores one1152-vector per layer and family/template pair:32 independent vectors after accounting for reversal, while retaining the full original network. This is not whole-model compression, a unique canonical form, or proof of a fundamental semantic unit.

The evidence now favors a transferable component of context-conditioned computation over arbitrary item-by-item operator lookup. Its demonstrated scope is narrow: two task families, two fixed contrasts, four held-out items per family and three previously studied probes. Candidate-versus-exact-token competence limitations from H640 remain unchanged. No loss or accuracy improvement is claimed.

Next freeze these learned shifts and test a genuinely new item set and new independent input probes without refitting or choosing templates based on outputs. Include the no-shift and other-family controls, exact context replacement as evaluation target, and both receiver/donor endpoint conditions. This tests prospective prediction rather than another retrospective fit. If it transfers, then investigate how few context coordinates are needed and whether shifts can be predicted from instruction structure across unseen contrasts.

Audits:512 cached attention-stage reconstructions;768 batched JVP calls; no new whole-model forwards; exact/no-shift replay against H640 is numerically identical; all source/runner/locked-model hashes unchanged; fitting indices restricted to items0–3; reverse shifts checked as negatives; finite metrics. No new natural-text documents; FineWeb remains26,112. Broad goal active and incomplete.
