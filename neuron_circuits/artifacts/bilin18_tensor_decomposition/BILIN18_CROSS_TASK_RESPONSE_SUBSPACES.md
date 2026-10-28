## H634 — Discovery response subspaces leave substantial held-out structure

H633 was progress: task-average responses transferred across items better than across wording, even when changed-wording copying was successful. H634 tests whether allowing multiple response directions can rescue a compact representation.

Use the frozen80-prompt, nine-direction H633 captures. Fit on20 discovery contexts (first four items of the first template in each of five tasks) and only five input probes: original, span0, span1, orthogonal0, orthogonal1. Hold out span2, span3, orthogonal2, orthogonal3 from basis fitting. This partition was declared after H633 outputs had been observed, so it is a retrospective holdout, not a blinded prospective replication. No new prompts or model evaluations occur.

Center output-logit derivatives under the pooled discovery token probabilities, then fit weighted SVD bases. The global fit has100 response rows; each task fit has20. Initial global ranks are1,4,16,32; task ranks1,4,16,20. After inspecting those results, add the global full100-row span as an exploratory ceiling check. Do not select a winning rank on test data. All bases use the same discovery-only probability metric.

For every test response, solve its best weighted least-squares projection under that receiver's native probabilities. **These coefficients use the true test response.** This is a representational upper bound for a frozen basis, not a predictive model, compressed language model, or canonical form. A future coefficient predictor can only do worse in this metric for the same basis. Native probabilities are also receiver-specific. Learned task labels select a task basis without using the test derivative; only projection coefficients are oracle.

**Allowing a subspace helps, but transfer remains incomplete.** A rank32 global basis has mean relative error0.352 for fitted input directions on new same-template items, versus0.572 for held-out input directions. Changed wording raises those to0.610 and0.635. A rank20 task basis leaves0.351/0.616 on same-template fitted/held-out directions and0.652/0.689 on changed wording. Thus favorable performance on reused directions substantially overstates generality to new directions.

| Test contexts | Input directions | Global rank4 error | Global rank16 | Global rank32 | Full discovery span100 | Mean energy captured at100 |
|---|---|---:|---:|---:|---:|---:|
|New items, original wording|fit|0.753|0.480|0.352|0.251|92.7%|
|New items, original wording|heldout|0.792|0.649|0.572|0.440|79.5%|
|Changed wording, all eight items|fit|0.825|0.696|0.610|0.481|74.9%|
|Changed wording, all eight items|heldout|0.840|0.710|0.635|0.502|73.2%|

Error is the receiver-probability-weighted norm of the centered residual divided by the true response norm. Energy captured is one minus squared relative error, averaged per example; it is not one minus squared mean error. Results average equally across five task families and input probes within each direction split. The changed-wording set contains both previously seen and held-out identities.

**The missing structure is not just the rank32 cutoff.** The full discovery span still leaves mean relative error0.440 on held-out directions in the original template and0.502 with changed wording. For explicit copying, where the changed template has100% top1 accuracy, even the full span leaves0.624 error on held-out directions and0.635 on reused directions. The full span recovers its actual fitting responses to below1e-14 relative error, so this gap is not failure to solve the projection.

This is evidence of incomplete coverage by the discovery response span, not proof that the model has no compact structure. Only five input directions and four examples per task/template informed the bases. More diverse discovery data, an internal representation, or a structured context-dependent operator could generalize better. A fixed output basis may be the wrong object because the final readout itself depends on the receiver's residual state. Generic random probes may also explore operations outside a small semantic task circuit.

Finite-dose KL diagnostics for the original direction at±0.1 are retained in the result JSON. They reconstruct logits with oracle projection coefficients and receiver-native logits. They are not an independent test of coefficient prediction; derivative error is the primary measure here. H633 already verified that the original receiver's own tangent predicts those finite doses accurately. This experiment leaves causal operator replacements and internal/readout separation untested.

Current hypothesis update: a single output vector or tiny fixed output basis is inadequate for these captures. Construction-dependent transformations remain plausible. A shared larger subspace is only partially supported: the global100-row span captures about79.5% of held-out-direction response energy on same-wording contexts and73.2% on changed wording. The remaining gaps are too large to call high-fidelity canonicalization. The particularly strong changed-wording copy failure cannot be dismissed as lack of native task competence. Failed plural/addition cells retain the limitations established in H633.

Next localize the gap. Capture internal residual tangents before the final readout, fit discovery-only internal bases under the same context/direction splits, then pass reconstructed tangents through each receiver's exact native readout. Compare this with the fixed logit-space bases. If internal reconstruction transfers while fixed output bases fail, readout geometry explains part of the apparent diversity. If internal reconstruction also fails, locate the earliest stage where held-out directions or constructions leave the discovery span. These tests should precede an expensive search for a universal low-rank representation; matched within-task versus cross-task factor replacements remain needed for causal interpretation.

Audits:source/manifest/source-result/runner hashes unchanged; no new forwards or natural-text documents;20 fitting contexts, five fitting directions, four direction holdouts; weighted basis orthogonality error below4e-14; normal-equation residual below3e-15; projection errors nonincreasing with nested rank; full-span fitting error below1e-14; all metrics finite. FineWeb remains26,112. Broad goal active and incomplete.
