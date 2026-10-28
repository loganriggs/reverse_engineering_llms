## H652 — Correcting malformed factual pairs preserves frozen-predictor transfer

**Correction to H651:** inspecting per-item failures exposed a construction error. The runner split the string containing South Korea on whitespace. This produced separate South and Korea subjects, shifted subsequent country–capital labels, and omitted Pakistan because zip truncated to eight answers. The resulting37.5% factual candidate accuracy was invalid. The claim that the predictor transferred across failed factual recall is withdrawn. H651 raw artifacts remain unchanged for audit; its reports and accumulated research notes carry an erratum. Its operator results describe malformed prompts, not the intended panel. Copy results were unaffected.

H652 builds explicit country and capital lists and checks every factual subject–answer pair in every form against the H651 pre-inference specification. The intended pairs are Australia/Canberra, Chile/Santiago, Cuba/Havana, South Korea/Seoul, Iran/Tehran, Iraq/Baghdad, Indonesia/Jakarta, Pakistan/Islamabad. No predictor, basis, mean, probe, regularization, or item-selection rule changes. All64 intended prompts are rerun and retained. Most prompts had already been evaluated in H651, so this is a repair of the prospective design, not a second fully independent blind replication. The frozen predictors still have no fitting access to these items or their responses.

**Corrected competence:** candidate accuracy is100% for both families in all four forms. Exact next-token accuracy remains variable:

| Family | Form0 | Form1 | Form2 | Form3 |
|---|---:|---:|---:|---:|
|Copy|25%|0%|0%|87.5%|
|Fact|62.5%|0%|87.5%|37.5%|

Candidate accuracy is selection among eight target tokens; exact accuracy asks whether the target is highest over the full vocabulary. The latter is not complete-response correctness. Answer-introducing words such as The can account for some exact-token failures. There is no candidate-incorrect subgroup to analyze in the corrected panel, so the proposed correct-versus-incorrect factual stratification is unsupported here.

**Frozen prediction still transfers.** The following error is measured relative to the exact local context-replacement effect using receiver-probability-weighted centered logit derivatives. No-change error is1. These are not accuracy or causal-share estimates. Oracle rank3 uses geometric projection of true context shifts, not output-optimal coefficients.

**Receiver incoming tangent/readout:**

| Condition | Mean8 | Oracle rank3 | Frozen predictor | Shuffled | Improved cases |
|---|---:|---:|---:|---:|---:|
|copy matched|0.381|0.326|0.359|0.386|53/64|
|copy broad|0.363|0.321|0.336|0.361|55/64|
|fact matched|0.463|0.431|0.428|0.499|60/64|
|fact broad|0.252|0.242|0.242|0.267|46/64|

**Donor incoming tangent/readout:**

| Condition | Mean8 | Oracle rank3 | Frozen predictor | Shuffled | Improved cases |
|---|---:|---:|---:|---:|---:|
|copy matched|0.384|0.329|0.365|0.389|55/64|
|copy broad|0.387|0.339|0.358|0.386|55/64|
|fact matched|0.462|0.428|0.426|0.497|58/64|
|fact broad|0.288|0.280|0.278|0.301|45/64|

The predictor improves all four group means over both mean8 and shuffled prediction at both endpoints. It improves214/256 receiver-endpoint and213/256 donor-endpoint cases. The copying results exactly reproduce the unaffected H651 results. Repeated prompt/probe measurements are correlated, not independent trials.

| Condition | Coefficient RMSE | Shuffled RMSE |
|---|---:|---:|
|copy matched|0.734|0.992|
|copy broad|0.595|0.998|
|fact matched|0.882|2.018|
|fact broad|0.317|1.556|

Zero correction has coefficient RMSE1. Matched factual coefficients are only weakly predicted geometrically, yet still give a functional improvement. Output fidelity and Euclidean coefficient reconstruction remain distinct objectives. Frozen coefficients can outperform the geometric oracle on output error because that oracle does not optimize outputs.

The supported hypothesis is unchanged: a stable template-conditioned operator component plus a smaller upstream-predictable correction transfers across items and directions within these two families. The withdrawn failure finding provides no evidence for generic instruction processing versus successful recall. This distinction must be tested directly with valid controls.

Next broaden operations before further optimizing copying/capitals. A useful design crosses the same item vocabulary with copying, a semantic relation, morphology, arithmetic, and syntax where feasible; uses matched instruction changes and separate wording shifts; and reports native competence before operator fits. Include explicit no-operation/format controls and split fit versus transfer items before output inspection. Refresh the hypothesis tournament to prioritize tests that separate generic formatting, lexical identity, task-specific transformation, and downstream readout explanations. Task labels remain supplied, and current methods retain the original model; they do not establish canonical fundamental units or end-to-end fidelity.

Audits: explicit manifest equality to the frozen intended pairs, all64 prompts retained, frozen predictor/probe/runner hashes unchanged, no fitting;576 whole-model forwards;512 cached attention-stage evaluations;3072 batched JVP calls (more length buckets after restoring South Korea). Native residual-chain maximum relative error below0.000430. All metrics finite. FineWeb remains26,112 unique documents. Previous goal turn was progress but included an erroneous competence interpretation, corrected here. Broad research goal remains active and incomplete.
