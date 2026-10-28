## H968 — Interaction trace reveals later amplification, not a uniquely localized composer

Previous turn classified as progress: exact MLP12 decomposition and independent direct evaluations established predominantly inherited interaction. This turn captures all18 blocks on the locked96-state factorial panel: block inputs, embedding skip x0, attention outputs, normalized MLP inputs, MLP writes and block outputs at the final token. Learned residual scales are explicitly included. Same-final-token alignment is functional; prompt lengths/absolute positions can differ.

For each quartet, exact four-state coefficients are computed. The final-token initial interaction and every x0 skip interaction are zero because the quartet ends in the same token. Raw pre-MLP12 interaction is reconstructed as the sum of earlier attention and MLP interaction writes multiplied by all intervening learned residual scales, plus attention12. Independent NumPy reconstruction agrees within1.19e-6 relative.

Largest signed projections onto raw pre-MLP12 interaction in the sole native-competent cohort (four familiar original-format quartets):

| Source write | Signed projection share |
|---|---|
| MLP11 |24.54%|
| MLP10 |17.44%|
| MLP9 |15.46%|
| MLP8 |9.34%|
| Attention11 |6.45%|
| Attention10 |5.98%|
| MLP7 |4.43%|
| Attention9 |3.57%|

These are additive native-trajectory bookkeeping shares, not intervention mediation fractions. Later writes are treated as their observed values in this identity; under an actual upstream intervention they would change. Thus a small early share does not imply weak causal importance. Attention writes can import composition formed at earlier token positions.

Exact local/inherited bilinear decompositions at all18 MLPs show local adjective-by-negation cross-product shares of0.37%,0.30%,0.46%,0.53% at MLP8,9,10,11 respectively, in signed projection onto each block's own output interaction. These large late contributions mostly transform already-present interaction. Raw interaction norm grows through these sites, but norm alone is not a task-importance metric.

The earliest block shows a different pattern: final-token attention0 interaction norm is only0.031 while MLP0 interaction norm is100.1; its local cross-product share is85.1%. However, learned scales and later geometry leave its direct additive projection onto raw pre-MLP12 interaction near zero (3.89e-7 share). This is a hypothesis-generating observation, not evidence that MLP0 computes semantic negation. Initial attention can produce nonsemantic factorial effects from context/position, and semantic relevance needs interventions. All later MLP local shares in this cohort are much smaller (MLP1 about5.47%, MLP2 about2.40%, then generally below2%).

Native competence restrictions from H966 remain unchanged: only familiar original-format cohort passes. Failed contexts are retained as diagnostics. No featurizer fitting, task-holdout training, or reserved-test evaluation.

Audit resolution: initial run failed a relative interaction-replay threshold; a diagnostic rerun showed worst error1.37e-4 relative to a small interaction extracted by subtracting large float32 writes. Native logits replay exactly, residual recurrence error6.85e-8 relative, normalization replay3.81e-6 absolute. Added two separate checks instead of silently relaxing that threshold: interaction mismatch relative to all output coefficients7.10e-7, and factorial interaction identity on double-precision reconstructed writes4.12e-14. Both pass. Preserve original interaction-relative error as a reported diagnostic. Audit attempts used192 forward examples; successful run96, total288. Original initial runner and failed diagnostic log remain in work/. Independent NumPy trace and share aggregation pass; finite outputs, sources unchanged; natural-document pool26,112.

Next: finite removal of the locked factorial interaction from attention and MLP writes across layers, with identity and norm-matched random controls. Recompute all downstream behavior naturally. This distinguishes early causal contributors from late amplifiers and avoids selecting only the largest native-trajectory write. If early attention is implicated, extend tracing across token positions. H961 frozen context-bridge predictor evaluation remains pending.
