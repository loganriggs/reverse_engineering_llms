## H823 — Encoder selection does not reproduce the observed local geometry under a Gaussian null

H822 found stronger local affine approximation in21 same-token early residual block clouds than Gaussian and radial controls, but those controls skipped the BSF's TopK gate. H823 directly tests that missing explanation. The previous goal turn made progress by completing H822 and identifying this specific confound.

### Frozen, selection-aware control

For each of the21 H822 token/block clouds, fit a Gaussian to ALL training occurrences of that token in the full1152-dimensional standardized residual, before block selection. Preserve the empirical training mean and sample covariance using a double-precision SVD factor; no diagonal covariance assumption or supervised labels. Pass generated activations through the unchanged signed linear encoder and its Top2 block-L2 gate. Decode the selected block into the same orthonormal16-dimensional coordinates used in H822.

For each of three seeded realizations, retain exactly the observed active training and validation counts. The first accepted points form the synthetic training set, the next the validation set. Draw no more than65,536 per realization and retain coverage failures. Compare the gated cloud with ungated block coordinates from the first same-count points in the identical Gaussian stream. Fit rank4 affine PCA globally or in32/64 nearest-training-point neighborhoods, using the same evaluation code as H822. This matches selected sample counts, not total encoder evaluations or native occupancy.

All63 cloud/seed realizations reach the required counts. The accepted run generates69,120 synthetic residual vectors; maximum draws per realization2,560. Acceptance ranges3.81%–100%, median99.80%;24/63 accept every draw. Most selected token/block combinations therefore activate almost automatically under this null, consistent with earlier strong lexical support. Low-acceptance cases are retained.

### Results

Numbers are median local/global reconstruction-error ratios across token/block clouds. Null ratios average three seeds within each cloud before the median. Lower ratios indicate stronger local approximation gains, not better causal fidelity or resource efficiency.

| Site | Neighbors | Clouds | Natural | Gaussian through gate | Gaussian without gate |
|---|---:|---:|---:|---:|---:|
| residual0 |32|8|.468|.839|.867|
| residual0 |64|8|.555|.956|.993|
| residual1 |32|13|.439|.814|.812|
| residual1 |64|13|.533|.945|.941|

All21 natural clouds beat their gated-null mean at32 neighbors. At64 neighbors20/21 do. These comparisons are descriptive and dependent, with only three null realizations; they are not21 independent discoveries or formal significance tests. They strengthen H822's evidence for non-Gaussian local structure beyond the specific explanation of selecting an otherwise Gaussian residual distribution with the trained BSF gate.

Gate changes are small for the median cloud: gated-minus-ungated error ratio at residual0 is -.00047/-.00090 for32/64 neighbors and exactlyzero at residual1. Differences of table medians are not medians of paired differences. Substantial truncation in a minority of clouds is compatible with these median results; all per-cloud records and acceptance rates are saved.

### What remains unresolved

The result does not distinguish smooth manifolds from mixtures of discrete contextual states, correlated position effects, density variation or non-Gaussian selection interactions. It does not prove that the trained blocks expose uniquely meaningful model units. Covariance estimation uses the same finite training token sample, can be rank-deficient, and has no shrinkage. The null omits higher-order activation structure by design. Natural validation was already used in checkpoint/block selection and prior experiments, so this is exploratory validation, not a fresh confirmation or reserved-test result.

Local affine approximations have more stored parameters and use the query to choose neighbors. Their lower geometric error is not evidence of a superior compact encoder. Early native activation geometry and L6 finite-response compression remain different experiments. Keep flat and shared-group featurizers as alternatives, and do not extrapolate a gate-null result to hierarchical or computational-DAG claims.

### Correctness and numerical repair

The first float32 covariance factorization failed the predefined covariance reconstruction tolerance: maximum relative error9.94e-4, above1e-5. An identical diagnostic replay exposed that failure; both outcomes were rejected and logs retained. Encoder gate replay was already exact. Rebuild covariance factors and Gaussian draws in float64, then cast generated residuals to float32 for the unchanged encoder. The accepted covariance reconstruction error is1.55e-12, gate/code replay error exactlyzero, all tracked source/checkpoint hashes unchanged, all metrics finite, and all coverage complete. Source configuration records the repair. No original-model forwards, no featurizer refitting, no reserved-test evaluations and no new natural documents; natural-document pool26,112.

### Next discriminating experiment

Now test whether local geometry predicts computation rather than only reconstruction. Use same-token, matched-position natural donor/recipient activations at the early residual sites. Compare near and distant donors while matching patch magnitude, and contrast selected-block edits with global PCA and matched random controls. Evaluate downstream distributional effects and donor-context transfer, keeping context clusters and continuous geometry as competing explanations. Preserve independence of donor selection from causal outcomes and use fresh natural occurrences or explicitly label reused validation. Before expanding the intervention batch, refresh the hypothesis set to include smooth geometry, discrete mixtures, amplitude, lexical/position confounds and downstream sensitivity as separate explanations. Retain the eventual diverse-competent-task requirement before assigning semantic or computational labels.
