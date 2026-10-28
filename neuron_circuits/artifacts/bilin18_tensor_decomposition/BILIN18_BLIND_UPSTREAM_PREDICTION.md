> **Correction — H652:** H651 split the multiword country “South Korea” on whitespace, producing malformed subjects and shifted factual answer labels. The factual competence figures and interpretation about successful versus failed recall are invalid and withdrawn. The saved derivative metrics describe the actual malformed prompts, not the intended factual panel. Copy results are unaffected. H652 repairs the pairing against the original preregistration with unchanged predictor/probes. Original files are retained for audit.

## H651 — Frozen upstream prediction transfers, including on poorly answered facts

H650 found useful rank3 correction coefficients predictable from the receiver residual leaving block9. H651 freezes that predictor, its shuffled-label control, and H648 means/bases before inference on a new64-prompt panel. It retains the same four templates per family, eight new copying items and eight new factual items. New copying words: cloud, beach, road, clock, phone, bread, mirror, train. New factual pairs: Australia/Canberra, Chile/Santiago, Cuba/Havana, South Korea/Seoul, Iran/Tehran, Iraq/Baghdad, Indonesia/Jakarta, Pakistan/Islamabad.

Several initially proposed capital names were multiple target tokens. The candidate list was expanded before inference, and the first eight eligible complete capital names were retained. No model outputs informed selection. The preregistration records the full candidate list, chosen items, frozen model and runner hashes, seed651, and analysis plan. Three new equal-norm directions are orthogonal to the previous twelve directions; the original anchor remains. All64 prompts and all four probes are retained. There is no coefficient fitting, basis update, or regularization selection on this panel. This is a prospective replication for new items/directions, not unseen tasks or templates.

**The frozen correction transfers.** Functional error is the receiver-probability-weighted centered logit-derivative error relative to the exact context-replacement effect magnitude; no-change error is1. This measures a local operator counterfactual, not task accuracy or a causal share.

**Receiver incoming tangent and readout:**

| Condition | Mean8 only | Oracle rank3 | Frozen upstream predictor | Shuffled labels | Improved cases |
|---|---:|---:|---:|---:|---:|
|Copy matched|0.381|0.326|0.359|0.386|53/64|
|Copy broad|0.363|0.321|0.336|0.361|55/64|
|Fact matched|0.438|0.408|0.407|0.465|56/64|
|Fact broad|0.267|0.259|0.259|0.279|39/64|

**Donor incoming tangent and readout:**

| Condition | Mean8 only | Oracle rank3 | Frozen upstream predictor | Shuffled labels | Improved cases |
|---|---:|---:|---:|---:|---:|
|Copy matched|0.384|0.329|0.365|0.389|55/64|
|Copy broad|0.387|0.339|0.358|0.386|55/64|
|Fact matched|0.443|0.411|0.411|0.470|57/64|
|Fact broad|0.292|0.288|0.286|0.303|37/64|

All four cell means improve versus both mean-only and shuffled-label prediction at both endpoints. Within each cell, each of the four probes also improves in mean error. Overall the correction improves203/256 receiver-endpoint and204/256 donor-endpoint prompt/probe cases. These are correlated repeated measurements, not independent statistical trials. Broad factual gains are especially small and less consistent across individual cases. Shuffled prediction slightly improves broad copying relative to mean-only, so that control is not uniformly harmful.

The oracle rank3 coefficients minimize geometric context error rather than output error; frozen prediction occasionally outperforms this comparison on functional error without implying donor-free access to an output-optimal solution.

| Condition | Coefficient relative RMSE | Shuffled coefficient RMSE |
|---|---:|---:|
|copy matched|0.734|0.992|
|copy broad|0.595|0.998|
|fact matched|0.789|1.885|
|fact broad|0.365|1.576|

Zero correction has coefficient RMSE1. All four correctly labeled predictions beat that baseline and the shuffled control.

**Native task competence limits the interpretation.** Candidate accuracy chooses among this panel's eight supplied target tokens; exact top1 asks whether the target is the model's most likely next token among the full vocabulary. Neither metric measures complete generated responses.

| Family | Form | Candidate accuracy | Exact next-token accuracy |
|---|---:|---:|---:|
|copy|0|100.0%|25.0%|
|copy|1|100.0%|0.0%|
|copy|2|100.0%|0.0%|
|copy|3|100.0%|87.5%|
|fact|0|37.5%|25.0%|
|fact|1|37.5%|0.0%|
|fact|2|37.5%|25.0%|
|fact|3|37.5%|0.0%|

Factual candidate accuracy is37.5% in every form, unlike the earlier panels'100%. All factual failures remain. The predictor's transfer therefore demonstrates repeatable instruction-conditioned operator structure even across poorly answered items; it does not establish that the recovered structure performs successful factual retrieval. Copying retains100% candidate accuracy but also has large exact-next-token variation. This observation makes task competence a discriminating axis for subsequent experiments, rather than a peripheral evaluation caveat.

The current leading explanation remains a stable template-conditioned component plus a smaller item-dependent correction available upstream. Alternative explanations include generic instruction/format processing, correlated lexical properties, and a mixture of task execution with failure/default-answer behavior. The present replication cannot distinguish these. To move toward fundamental computational units, the next experiment should test whether the same description changes with success versus failure and whether analogous structure holds across additional operations, rather than further refining these two task families alone. Begin by auditing per-item factual predictions and stratifying the frozen-effect results by native candidate correctness, without dropping failures or treating post-hoc subgroups as confirmatory. Then construct a broader task panel with competence and template changes crossed explicitly.

This retains the original network and known family/template identity. It predicts local context-replacement effects, not finite behavioral edits, output KL under full model replacement, or a unique canonical form. Residual errors remain substantial.

Audits:576 whole-model native/finite-difference forwards;512 cached attention-stage evaluations;2304 batched JVP calls; no predictor refit. Frozen preregistration hashes and all runtime hashes unchanged. Native residual derivative chain agrees with central-difference captures within0.000430 relative error. Metrics finite; matched template token lengths equal. New probes orthogonal to all twelve prior directions to below1e-12. No new natural-text documents; FineWeb remains26,112. Broad research goal active and incomplete.
