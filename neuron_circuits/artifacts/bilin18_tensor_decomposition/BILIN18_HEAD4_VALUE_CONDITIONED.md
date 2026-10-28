## H665 — Mean routing change applied to receiver values explains much of head4's item variation

H664 localized layer14 head4 approximation error to input-token contributions. H665 factors the input write into signed routing coefficients c and projected effective value vectors V. For aligned input positions, the true copy→task change is sum(c_task V_task − c_copy V_copy). Copy/task input-token ID sequences are required to match exactly. Every input token is included; coefficient means are fitted per ordinal position, with fixed input-token count checked within each family. Four second-wording discovery items fit each family/form model; no held-out donor factor enters the predictive methods.

The new predictor replaces the unconditional input-write mean with sum(mean_discovery(c_task−c_copy) × V_receiver). It uses the receiver's actual values, preserving input identity, but only a discovery mean for the routing change. All other head-role, attention-layer and boundary predictions remain fixed. An alternative fixed-product control uses the product of discovery mean coefficient change and discovery mean receiver value, removing item-specific receiver values while keeping the factorized form.

Two donor-informed controls isolate factors: exact coefficient change applied to receiver values, and receiver coefficients applied to exact value changes. Exact input-write replacement remains the reference. These partial controls are not an additive decomposition of the full effect: the identity includes the interaction term Δc ΔV, alongside Δc V_receiver and c_receiver ΔV. Effective values are not assumed context invariant.

**Predicted boundary, finite KL(task||edited):**

| Family | Unconditional mean | Mean coefficient × receiver value | Fixed product of means | Exact coefficient × receiver value | Receiver coefficient × exact value change | Exact input write |
|---|---:|---:|---:|---:|---:|---:|
|addition|0.007949|0.007606|0.007946|0.007298|0.007957|0.007310|
|agreement|0.005285|0.005172|0.005283|0.005238|0.005259|0.005242|
|antonym|0.011206|0.010064|0.011179|0.009873|0.011241|0.009891|
|capital|0.017518|0.011400|0.017684|0.010051|0.017113|0.009953|
|plural|0.010843|0.010455|0.010944|0.010187|0.010887|0.010121|
|successor|0.086528|0.022530|0.086469|0.032363|0.075352|0.032417|

The donor-free mean-coefficient predictor improves every family mean, with the largest effect for succession:0.08653→0.02253. The fixed-product control remains0.08647, so merely rewriting the mean as a product does not produce the gain; retaining the receiver's item-specific value matters. Capitals improve0.01752→0.01140. Gains for addition, agreement and plurals are smaller. No hyperparameter or output-loss fitting occurs.

For succession, the predictor is a particularly simple rule: one mean signed coefficient change per known form multiplied by the receiver's projected month-token value. This is a concrete candidate for a conditional computational description, rather than a fixed write vector averaged across month identities. It still depends on the original model's receiver state, known family/form and preselected head; it is not a complete month-succession algorithm or global semantic unit.

**Exact boundary comparison:**

| Family | Unconditional mean | Mean coefficient × receiver value | Exact coefficient × receiver value | Exact input write |
|---|---:|---:|---:|---:|
|addition|0.001671|0.001503|0.001352|0.001357|
|agreement|0.001706|0.001614|0.001655|0.001660|
|antonym|0.009779|0.007804|0.007618|0.007631|
|capital|0.013138|0.007941|0.005881|0.005811|
|plural|0.007369|0.007084|0.005340|0.005283|
|successor|0.081667|0.020535|0.021853|0.021882|

For succession the mean-coefficient predictor slightly beats the exact-input repair on total output KL. This does not make it a more accurate reconstruction of the input write. Other boundary/head errors remain, and an inexact repair can compensate for them. Exact component replacements are references, not output-optimal bounds.

**Geometric input-write relative RMSE:** sqrt(sum squared prediction error / sum squared true write-change norm).

| Family | Unconditional mean | Mean coefficient × receiver value | Exact coefficient × receiver value | Max receiver→task relative value change |
|---|---:|---:|---:|---:|
|addition|0.7963|0.5792|0.4360|0.1325|
|agreement|1.0767|0.7632|0.3062|0.0886|
|antonym|1.0330|0.6492|0.3425|0.0684|
|capital|1.0724|0.4355|0.0887|0.0686|
|plural|1.0450|0.9569|0.4768|0.0623|
|successor|1.2157|0.4196|0.0234|0.0905|

Succession's unconditional mean has geometric error1.2157; conditioning on receiver values reduces it to0.4196. Exact coefficients on receiver values reduce it further to0.0234, despite receiver/task effective value differences up to9.05%. Thus routing changes acting on existing receiver values dominate this particular input-write change. This does not mean values are universally fixed or irrelevant. Addition/plural exact-coefficient errors remain substantially larger, and the analysis should not be generalized indiscriminately.

The new rule targets the most useful missing dependency exposed by the earlier diagnostics: input identity in the value vector. It explains why averaging whole writes across identities can fail even when a simpler coefficient-level change transfers. This supports conditional units of the form routing × content as a research direction. The bilinear factorization is architectural; the empirical finding is that a discovery mean coefficient change is useful on new items within these particular templates.

Fidelity and behavior remain separate. All conditions still have0% exact next-token correctness on this held-out subset. The task reference distribution is frequently incorrect, and candidate accuracy is not full-response accuracy. The current panel has been repeatedly inspected; discovery-only fitting does not make this an independent prospective replication. The predictor's head, role and form were selected through prior diagnostics.

Next freeze the coefficient means, boundary/attention means, token-role rule and evaluation metrics. Test a genuinely fresh item panel across all six families with the same wording, including September–December for month succession, before further fitting. Keep tokenization-based eligibility checks separate from output-based selection. Compare unconditional mean, receiver-value-conditioned prediction and fixed-product control, with exact-factor controls only for interpretation. New wording generalization should remain a distinct later test. A successful replication would support a more stable conditional description; failure would reveal limits currently hidden by repeated analysis of one panel.

Audits:1152 whole-model forwards (2 boundary choices×6 methods×96 prompts),1152 cached attention-stage evaluations/reconstructions. Native head and role-residual reconstruction errors remain below4.81e-7 and7.06e-6 relative. Baseline and exact-input repair logit arrays reproduce H664 identically. Aligned complete input-token sequences and within-family ordinal counts asserted; all hashes unchanged; finite metrics. No held-out factor in the mean-coefficient or fixed-product predictors, no output fitting. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
