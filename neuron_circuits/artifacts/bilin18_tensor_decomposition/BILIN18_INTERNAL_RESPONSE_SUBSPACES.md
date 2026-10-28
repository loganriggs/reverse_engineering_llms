## H635 — Residual passthrough explains early direction-holdout error, but internal computation remains context-dependent

H634 was progress: discovery-only output subspaces left substantial error on held-out directions and wording, even with oracle coefficients. H635 asks whether this is an artifact of ignoring the residual identity path or of the receiver-specific final readout.

Reuse the frozen five-task80-prompt panel, nine probes and retrospective five-fit/four-held-out direction split. Run native and±0.02 interventions for every probe, capturing final-position residuals after blocks5,9,13,17. This requires1520 model forwards. Replayed native logits and derivative logits match H633 exactly. The exact native readout derivative applied to measured block17 residual tangents matches measured logit derivatives within0.000904 relative weighted error.

For each stage, fit global Euclidean SVD bases using only20 discovery contexts times five discovery probes. Fixed ranks4,16,32,100. Compare two representations: the full residual tangent, and the additional response after subtracting the exact direct residual path. For injection after Attention5, that direct contribution is q at block5, then q times the product of subsequent block residual lambdas. Scale factors at blocks5/9/13/17 are1.000,0.5054,0.5453,0.8300. We call the remainder “innovation” only as an algebraic definition: it includes all attention/MLP response contributions and their interactions along the derivative chain, not an identified semantic feature.

Reconstruct the innovation with oracle Euclidean coefficients and restore the known direct path. Report error relative to the full true tangent so both methods are comparable. The direct path is known from the injected direction and weights; it is not fitted using held-out responses. Basis coefficients do use true test responses, so these remain representational tests rather than predictive models.

**The identity path is a major early confound for input-direction holdout.** At block5, full-span100 mean residual error for held-out directions is0.920 with original wording and0.923 with changed wording. Restoring the direct path lowers these to0.545 and0.561. Thus much of the early error comes from trying to express new orthogonal input directions in a basis learned from other directions. It should not be interpreted entirely as context-specific feature computation.

**Substantial missing structure remains after that correction.** For held-out directions, full-span100 innovation reconstruction errors rise through blocks9 and13 before falling at block17. The fall is descriptive: each stage has a separately fitted basis and its own denominator, so this is not a causal accounting of information creation or destruction.

| Context wording | Block5 full / skip-restored | Block9 full / skip-restored | Block13 full / skip-restored | Block17 full / skip-restored |
|---|---:|---:|---:|---:|
|Original wording, new items|0.920 / 0.545|0.843 / 0.784|0.798 / 0.769|0.550 / 0.539|
|Changed wording|0.923 / 0.561|0.897 / 0.847|0.877 / 0.854|0.540 / 0.533|

Reused input directions also show construction sensitivity after correcting the direct path: block5 mean error grows from0.137 on same-wording new items to0.464 with changed wording; at block9 the corresponding errors are0.326 and0.796. This is not explained by holding out an orthogonal injected direction. Four coarse stages were captured; no claim is made about the exact earliest token or sublayer that creates these differences.

**Euclidean residual fidelity and output fidelity differ.** Mapping Euclidean block17 reconstructions through each receiver's exact native readout yields held-out-direction errors0.694/0.797 at rank100 with skip restoration. These exceed fixed-output-basis errors from H634. That alone would be misleading: Euclidean coefficients are not optimized for the output metric.

A supplemental audit therefore maps each fixed internal basis through the receiver-native readout and solves the best receiver-probability-weighted output projection. For innovation bases, subtract the known direct-path output before fitting, then restore it. This uses oracle test responses and exact receiver readout; it gives the favorable representational ceiling available within the tested affine space. The audit was added after inspecting the Euclidean results, not selected from alternative bases.

| Contexts / directions | H634 fixed output basis100 | Internal full basis100, output-optimal | Internal innovation100 plus direct path, output-optimal |
|---|---:|---:|---:|
|Original wording / fit|0.251|0.249|0.249|
|Original wording / heldout|0.440|0.435|0.426|
|Changed wording / fit|0.481|0.474|0.470|
|Changed wording / heldout|0.502|0.489|0.482|

Receiver-specific readout and the direct-path correction give only modest improvements over H634's output basis at the full discovery span. They do not rescue high-fidelity transfer: held-out-direction errors remain0.426 with original wording and0.482 with changed wording. Fully competent changed-wording copying still has0.599 error. Consequently, final-readout geometry alone is not a sufficient explanation for these discovery-coverage failures. This conclusion is limited to the tested spans and oracle projection methods, not all possible structured representations.

The main update is twofold: preserve the known identity path when evaluating novel-direction generalization, and avoid interpreting Euclidean residual reconstruction as optimal output preservation. After both controls, internal response variation remains substantial, including variation across constructions for the same injected directions. A universally shared fixed response basis is not currently supported at high fidelity. A context-conditioned operator remains a more plausible object to investigate than repeatedly increasing a fixed output basis's rank.

Next return to mechanistic contrasts rather than more rank searches. Use matched item pairs across the two copying/factual/agreement constructions, retain the inflection/arithmetic failure controls, and decompose changes in normalized-MLP responses into bilinear operating-context and normalization contributions. Compare matched within-construction references against cross-construction references, preserving direction and item identity where possible. Propagate decisive local contrasts to output with native downstream operators. This can test whether construction-dependent operators explain the newly localized differences, rather than only documenting lack of compression. Unseen-direction and held-out-item checks remain required before a transferable unit can be claimed.

Audits:all input/runner hashes unchanged;1520 forwards; exact native/derivative replay against H633; final-readout relative validation error below0.001; internal basis orthogonality below8e-14; supplemental output-optimal normal equations below2e-19; finite metrics. Supplemental script and result are separately hashed. No new natural-text documents; FineWeb remains26,112. Task competence limitations and retrospective direction split carry over unchanged. Broad goal active and incomplete.
