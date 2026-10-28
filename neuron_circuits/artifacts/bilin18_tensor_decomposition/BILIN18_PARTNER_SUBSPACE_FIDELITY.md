## H885 — Native-fidelity testing reveals task-specific partner structure and a copying failure

H884 showed that context means can make responses agree without preserving native computation. H885 directly compares approximate and live interventions. A task-specific mean approximates human-name-binding task responses well; agreement benefits substantially from a learned variation direction; language recall retains more vocabulary-wide error. The same approach fails badly on copying those names. This is evidence for computation-dependent structure, not a universal partner manifold or a completed featurizer.

### Estimation and controls

Retain H884's first-four discovery / last-four transfer pairs per domain. For each domain, family, template and MLP layer L8–17, fit a centered PCA basis to the eight discovery endpoint vectors after native RMS normalization. Evaluate fixed ranks 1, 2, 4 and 7; rank 7 spans the centered discovery data. Include a random rank-7 orthonormal subspace, own-context mean, mean pooled over task/copy families, and wrong-family mean. All ranks and controls are fixed before transfer outcomes; no rank is selected on transfer performance.

Approximate the recipient's normalized native partner vector by discovery_mean + projection(native_partner−mean), without renormalizing. Recipient-native normalization and output offset remain intact. The linearized MLP response uses this approximated partner vector; attention and unselected computation remain live. Coordinates use the recipient's own native activation, but fitting uses discovery only. This is local partner-response compression with native context access, not compression of the whole model, an unsupervised task classifier, or reconstruction of all native outputs from a compact code.

All 192 contexts run under identity and symmetric headwrites. Live and recipient-native linearized controls distinguish partner approximation from the finite-nonlinearity approximation. Transfer pairs were previously measured in H883–884, so these are estimation-disjoint checks, not untouched confirmation. Each domain has only four transfer pairs.

### Transfer fidelity: native task-context readout

Individual errors retain both intervention signs, endpoint orders and formats before comparison. They measure deviation from the same live intervention, rather than task/copy mismatch.

|Domain|Mean: individual answer-response error|Rank 1|Rank 2|Rank 4|Rank 7|Random rank 7|
|---|---:|---:|---:|---:|---:|---:|
|agreement|1.497|0.452|0.513|0.416|0.445|1.489|
|language|0.168|0.185|0.182|0.172|0.179|0.168|
|binding|0.057|0.056|0.058|0.057|0.056|0.056|

Agreement's learned rank-1 direction improves error from 149.7% to 45.2%, whereas random rank 7 remains at 148.9%. The response is weak, so relative errors require that qualification, but the improvement is not reproduced by adding arbitrary dimensions. Higher rank is not uniformly better for the selected margin. Language's answer margin does not improve with rank, even though vocabulary-wide fidelity does. Binding's mean already achieves 5.7% individual error; additional ranks have little margin benefit. Recipient-native linearization errors are 1.3%, 4.3% and 3.0%, respectively, providing the finite-response approximation reference.

### Direct vocabulary-wide fidelity

Compute KL(live edited distribution || approximated edited distribution) for each individual signed intervention. Normalize the summed KL by summed KL(live edited || unedited native). A ratio below one beats using the unedited output as the approximation under this metric. It is not a percentage of semantics explained or a guarantee of task success.

|Domain, task contexts|Own mean KL ratio|Rank 1|Rank 2|Rank 4|Rank 7|Random rank 7|
|---|---:|---:|---:|---:|---:|---:|
|agreement|0.350|0.269|0.239|0.219|0.205|0.338|
|language|0.593|0.581|0.561|0.460|0.393|0.585|
|binding|0.037|0.036|0.035|0.032|0.032|0.037|

Task-context mean versus rank-7 absolute fidelity KL is approximately 1.02e-5→5.98e-6 for agreement, 0.00118→0.000781 for language, and 0.000193→0.000166 for binding. Wrong-context means give substantially larger task KL ratios: 1.695, 0.802 and 0.437. The binding result is therefore context-specific rather than any arbitrary shared mean working equally well.

Copying contexts are much less forgiving. Their mean / rank-7 KL ratios are agreement 0.826/0.410, language 1.168/0.935, and binding 3.975/3.373. For copying names, both the mean and the entire seven-dimensional discovery span are worse than leaving the output unedited under the stated distribution metric. This prevents a universal compression claim. Full results retain both source and task-answer contrasts and the controlled readout separately from native readout.

### Interpretation and next test

The evidence favors different representation requirements for different computations: binding task contexts have a substantial reusable context component; agreement needs at least some context-internal variation; copying preserves lexical distinctions not captured adequately by these tiny discovery sets. Rank 7's training reconstruction is exact by construction and does not establish seven-dimensional intrinsic structure.

Next inspect the agreement rank-1 variation and the copying residuals. Test whether the discovery-derived agreement direction tracks grammatical number across unseen nouns and templates, and intervene on its coordinate while preserving the context mean. Use label association only as an interpretation diagnostic, with causal prediction evaluated on transfer pairs. For copying, distinguish insufficient discovery coverage from a genuinely different representation requirement using a fixed larger lexical pool and a learning curve, without selecting dimensions from the transfer outcomes. These tests connect the low-dimensional response approximation to meaningful computation, rather than naming a principal component a feature because it improves a metric.

### Verification

Completed 5,952 full-model forwards. Identity and H883 live/self reference final states replay exactly. Basis orthogonality error is at most 1.15e-14; rank-7 discovery reconstruction relative error 5.77e-15. Independently checked all 120 fit supports use only discovery endpoints. Analytic Jacobian versus autodiff relative error 3.36e-16, versus finite differences 1.08e-08; all ten layers checked. Selected readout error 8.43e-06. Identity fidelity KL max absolute 0; all evaluated fidelity KL values finite. Source hashes unchanged; no amplitudes dropped. No reserved test evaluated; natural-document pool unchanged at26,112. Broad goal active.
