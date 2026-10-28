## H833 — Lexical/context separation helps prediction, but unseen-token transfer worsens

Extend H832's training-only state benchmark using the same512 development-validation documents. No new states or full-model forwards. Fit contextual residual bases after subtracting each training token's normalized-state mean. Compare ordinary residual PCA with a response-weighted linear basis at ranks4/16/64; retain the same9016-token table and exact one-scalar radius access in every candidate.

### Response-weighted surrogate and correctness

For eight independent orthonormal fit probes v (seed833+layer, different from H832 evaluation probes), the bilinear response as a function of normalized state s has linear map T_v=D[diag(Rv)L+diag(Lv)R]. Set W=mean(T_v^T T_v)+ridge*I, with ridge=.001 times the mean diagonal. Fit PCA to W^1/2 times the training token residual. Encoder E=W^1/2 B and decoder F=W^-1/2 B give reconstruction x E F^T. This optimizes a regularized response-weighted state reconstruction surrogate. It excludes the full state-dependent RMS derivative and final shape renormalization during fitting; evaluation includes both. It is not direct optimization of full-model KL or the exact local response objective.

All fit data are training-only. Evaluation uses the H832 eight independent probe directions and512 documents, already designated development validation. Shape is reconstructed as token mean plus low-rank contextual residual, renormalized, restored with recipient radius and passed to the full analytic RMS/bilinear derivative. No true baseline derivative is supplied.

Linear-map equivalence error5.24e-15, weighted encoder/decoder duality error2.00e-15, and true-response squared norms replay H832 exactly. Input/model hashes unchanged and results finite. Fits, per-document/probe records and independent fit probes are saved. No full-model/test evaluations or new natural documents; natural-document pool26,112.

### Overall response prediction

Relative RMSE; zero response would score1:

| Site | Rank | Global PCA + radius (H832) | Token-residual PCA + radius | Response-weighted token residual + radius |
|---|---:|---:|---:|---:|
| block1 |4|.924|.810|.802|
| block1 |16|.857|.748|.743|
| block1 |64|.729|.644|.640|
| block2 |4|.957|.874|.865|
| block2 |16|.894|.817|.805|
| block2 |64|.745|.694|.687|

Separating lexical state from contextual residual produces the larger aggregate gain. Response weighting adds a small further improvement at all tested ranks/sites. Paired document-bootstrap2048-resample intervals for weighted-minus-residual-PCA error exclude zero at each setting, with gains roughly.004–.012. These are descriptive development comparisons, conditional on fixed probe sets; no multiplicity or probe-seed uncertainty correction is claimed. Absolute errors remain large, so none of these models is yet an adequate local computational description.

These are NOT fully resource-matched architecture wins. Global PCA lacks the9016-by1152 token table. Residual PCA and weighted residual share that table and r contextual coordinates plus radius, but weighted encoder and decoder differ and require2*1152*r stored entries, versus a shared1152*r PCA basis. Evaluation still uses full fixed MLP weights. Active-coordinate count is not full runtime or storage cost.

### Unseen-token breakdown changes the interpretation

Posthoc stratification of the512 documents by whether their sampled token appears in training positions gives403 seen and109 unseen tokens. “Unseen” concerns sampled featurizer training positions, not language-model vocabulary/pretraining. Unseen tokens use the predefined global-mean fallback in all token-conditioned models.

At rank64:

| Site | Token stratum | Global PCA error | Residual PCA error | Weighted residual error |
|---|---|---:|---:|---:|
| block1 | seen |.726|.570|.567|
| block1 | unseen |.737|.803|.797|
| block2 | seen |.747|.649|.643|
| block2 | unseen |.739|.826|.815|

The aggregate gain is concentrated in tokens with learned lexical means. For unseen tokens, contextual residual bases perform worse than global PCA. Removing known-token means during training reallocates capacity away from variation needed when the lexical mean is unavailable. This is a plausible interpretation, not a separately isolated mechanism. It directly cautions against treating aggregate reconstruction/response gains as reusable feature discovery.

A routing rule that uses global PCA for unseen tokens is now a motivated candidate, not a validated result. Its proposal is informed by these validation outcomes and must be frozen before confirmation. Token counts may matter for rare seen tokens as well; do not choose a threshold retrospectively and call it independent validation.

### Next step and broader scope

Freeze the current global, token-residual and response-weighted fits and a simple seen/unseen fallback rule, then evaluate on remaining unused documents with new probe directions. Report seen/unseen and overall metrics, not just the favorable aggregate. This tests transfer of the operator-state approximation rather than adding another flexible model to the same panel.

The exact-site BSF/shared-dictionary comparison remains outstanding. It should receive the same lexical/context decomposition and scale access as competing baselines, with explicit parameter and active-coordinate budgets, followed by downstream KL checks. Current evidence favors separating lexical and contextual state for this prediction problem, but does not demonstrate a better manifold ontology, hierarchy or DAG. Reusable computation must survive new lexical states and task combinations, not just improve mean error on familiar token identities.
