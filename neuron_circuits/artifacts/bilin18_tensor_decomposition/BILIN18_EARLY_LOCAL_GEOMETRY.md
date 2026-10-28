## H822 — Early token-conditioned blocks have local structure beyond simple elliptical clouds

Return to the user's early-layer manifold question. H684–686 found partially stable BSF blocks, strongly lexical support, contextual within-token coordinates, and some donor-context transfer under same-token block swaps. H687 onward implemented alternative architectures, but synthetic identifiability failures and later task-dependent fidelity mean no architecture is established as universally ideal. H821's completed flat controls reinforce retaining flat and grouped alternatives.

### A direct geometry test

Use all24 previously selected H685 width16 blocks: eight each from value0, residual0 and residual1, seed681. Freeze checkpoints and selection. Restrict each point cloud to active occurrences of one exact token in one block. Require64 training occurrences and8 validation occurrences. Training uses the original2048 documents and validation the following256; no reserved test is read. Validation was used in earlier selection, so this remains exploratory rather than fresh confirmation.

Map block codes into orthonormal coordinates of their decoded residual contribution using QR. Euclidean distances then respect the decoder geometry, even when decoder rows are nonorthogonal. Fit a rank4 affine PCA globally to each training token cloud. For each validation point, also fit rank4 affine PCA to its32 or64 nearest training points. Evaluate squared reconstruction distance to each affine plane. The query determines its neighborhood but does not enter the PCA fit.

This is adaptive geometric approximation, not prediction of an unobserved coordinate or a deployable compressed encoder. Local planes use many more stored parameters than one global plane. A local improvement alone therefore does not prove a manifold.

Calibrate against five Gaussian realizations per cloud with the empirical training mean/covariance and identical sample counts and neighborhood procedure. After inspecting those results, add a clearly posthoc radial control: random elliptical directions rescaled to each observed training/validation point's distance from the training mean. This preserves the radial magnitudes but not exact covariance. The initial Gaussian-only output is retained in work/h822_initial_gaussian.json.

### Results

Report the median local/global squared-error ratio across eligible token/block clouds; lower is better. Gaussian and radial columns are medians of the five-realization mean ratios for each cloud, not formal significance thresholds.

| Site | Neighbors | Clouds | Observed | Gaussian | Radial control |
|---|---:|---:|---:|---:|---:|
| residual0 |32|8|.468|.808|.821|
| residual0 |64|8|.555|.952|.934|
| residual1 |32|13|.439|.770|.743|
| residual1 |64|13|.533|.909|.846|

At32 neighbors, all21 eligible residual token/block clouds have lower local/global error than both control means. At64,19/21 beat Gaussian and20/21 beat radial controls. Therefore the locality advantage is not reproduced by these simple elliptical distributions, even when observed radial magnitudes are retained. The same data's Gaussian ratios below1 illustrate why an uncontrolled nearest-neighbor gain would be misleading.

Nine otherwise count-eligible value0 token/block clouds are degenerate: within-token training variance divided by mean squared norm is below1e-10. All are reported and excluded from ratio calculations. This matches value0's architecture-based lack of contextual variation and prevents numerical noise from masquerading as a manifold. Residual0 block111 has no count-eligible token cloud; coverage is retained rather than silently replacing the block. The21 clouds are dependent and cover a selected lexical subset, not21 independent discoveries.

### Interpretation and competing explanations

This provides evidence for non-Gaussian local geometry inside early token-associated blocks. It does not distinguish a smooth curved manifold from discrete context clusters, nonlinear density, or selection effects. In particular, active-block selection by the BSF's TopK encoder itself truncates the distribution, while the controls do not replay that selection. The featurizer could therefore induce some of the geometry being measured. Position and document context are not matched within these clouds. A local tangent estimate should not yet receive a semantic label.

The radial result weakens a pure magnitude explanation but does not eliminate it: the radial null is only one angular randomization with approximate covariance. The current evidence supports further testing of local structure, not superiority of a BSF over the newer shared-dictionary alternatives. The task-response basis results at L6 concern a different activation site and intervention family and must not be treated as directly matched architecture controls here.

### Verification and next discriminating test

Zero new model forwards or training, no new documents, reserved test untouched; natural-document pool26,112. All source and checkpoint hashes unchanged. Maximum decoded QR reconstruction error2.22e-15. All recorded ratios finite. Independent numerical checks verify exact affine rank4 reconstruction below1e-24 and invariance of nearest-neighbor PCA errors under rotation and translation within1e-10.

Next test whether the measured local advantage survives the encoder's selection mechanism: generate token-conditioned Gaussian residual activations from training statistics, run them through the unchanged BSF, and compare selected decoded clouds with identically processed natural data. Preserve finite sample/occupancy effects and report failures to obtain matched coverage. Then compare neighborhood geometry with donor-transfer effects on independent same-token occurrences, retaining global PCA, random controls and alternative featurizers. A useful computational manifold should predict intervention consequences, not merely local reconstruction. Diverse competent task panels remain necessary before naming computational units or adding a hierarchy/DAG.
