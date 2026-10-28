## H482 — Learning omitted-response directions with a capacity control

Following H481's oracle remainder result, test whether a few learnable output directions help. Hold the256 input coordinates, original768 kernel support, H4781024 training receivers/256 validation receivers, donor assignments, Fisher regression and ridge grid fixed. Compare the original16 output directions (base), those16 plus four training-response-remainder singular vectors (pca), and those16 plus four seeded Gaussian directions projected orthogonal to the original span (random). The pca/random arms both have20 output dimensions and5120 learned coefficients; base has4096. This controls added dimension between the augmented arms, not generalization across random bases.

Compute the native training response remainder after Euclidean projection into the original16-dimensional span, using both signs. Use uncentered SVD of that remainder: zero-change response is the reference, with no learned response intercept. Take the leading four right singular vectors, reproject and QR-orthonormalize. Random directions are independent of outcome data, seed20271206, also projected and orthonormalized. Neither added basis uses validation or confirmation targets. Do not assume remainder-energy ranking is probability-optimal.

Each arm fits the same anchored Fisher-weighted normal equations, selecting ridge from [1e−6,1e−4,.01,.1,1,10,100] by pooled finite validation KL with the original pre-refit predictor as fallback. Padding the original coefficients with zeros gives the same functional anchor in each basis. Penalties use each arm's mean Hessian diagonal, so numerical ridge values are not identical functional regularization across arms. The combined24-row storage basis contains both additions but each fit uses only its own16/20 rows; joint orthogonality of pca and random additions is neither needed nor claimed.

Freeze selected maps before collecting outcomes on512 new complete-document-hash-disjoint receivers. Independent audit rehashes15808 excluded documents and the512 new documents against their original caches, verifies all129-token windows and finds zero overlap. Total tracked documents16320. Freshness is within this continuation, not a model-training holdout or new donor family. The validation panel has adaptive history.

Selected candidates: {'base': 'base_0.1', 'pca': 'pca_0.01', 'random': 'random_0.1'}.

| Predictor | Fresh confirmation KL recovery −/+ | Raw recovery −/+ |
|---|---|---|
|original|73.33% / 74.45%|92.87% / 92.73%|
|base|75.04% / 76.71%|90.64% / 90.58%|
|pca|74.80% / 73.12%|93.91% / 93.75%|
|random|74.17% / 75.38%|90.76% / 90.69%|

Recovery is one minus pooled response prediction error divided by pooled native effect, not accuracy or average per-case recovery. Native intact state is retained for response isolation; no intact-state or observed-token CE claim is tested.

| Predeclared primary comparison | Gain | Joint95% interval | Lower>0 |
|---|---:|---|---|
|pca_minus_base_negative|-0.25pp|[-2.17, +1.67]pp|False|
|pca_minus_random_negative|+0.63pp|[-2.02, +3.28]pp|False|
|pca_minus_base_positive|-3.60pp|[-8.79, +1.60]pp|False|
|pca_minus_random_positive|-2.27pp|[-6.49, +1.95]pp|False|

0 of four primary gates pass. The training remainder carries 1.98% of total response energy; its top four singular directions capture 31.55% of remainder energy. These are training geometric quantities, not probability-error bounds.

Negative effects: effective count47.07/512, largest-case share10.22%, top26 share54.84%. These describe concentration, not alternative statistical sample sizes.

Positive effects: effective count31.32/512, largest-case share12.74%, top26 share62.34%. These describe concentration, not alternative statistical sample sizes.

Four simultaneous standardized bootstrap bands use4000 paired resamples within64 fixed donor strata, eight receivers per donor, shared across arms and signs. They condition on training/validation results, calibration and the single frozen random basis. They do not estimate variability over training samples or random-basis draws. Failed superiority is not equivalence.

Audits: each active basis orthonormal within1e−10; all regularized normal-equation residuals below1e−7; original map projection/whitened kernel coordinate audits pass; finite arrays; base validation grid independently replays H479 within1e−7 absolute KL. Candidate coefficient matrices, union output basis, residual spectrum, encoder, selection and per-case outcomes are saved. Complete-document/window audit passed independently. All model runs terminal. The broader feature/computation objective, CE comparison, source rank1/tail behavior and cross-head/layer transfer remain open.

Interpretation: all four predeclared improvement gates fail. PCA augmentation versus base changes KL recovery by−.25/−3.60pp, with joint intervals[−2.17,1.67]/[−8.79,1.60]pp; versus the single random control the changes are+.63/−2.27pp, intervals[−2.02,3.28]/[−6.49,1.95]pp. These intervals do not establish benefit or harm. Raw response recovery rises from90.64/90.58% to93.91/93.75%, while finite KL point estimates decline. That directly reinforces the distinction between internal Euclidean fidelity and behavioral fidelity; it does not show that all data-derived output augmentation is ineffective.

Training KL recovery improves substantially with the PCA addition, and validation selects it through the predeclared ridge grid, but the gain does not carry to this confirmation panel. This pattern leaves competing explanations: the added directions are poorly aligned with held-out probability sensitivity; their useful coefficients generalize poorly; adaptive validation plus concentrated outcomes gives a noisy choice; or the baseline local-Fisher objective does not capture finite effects. One fixed random basis cannot resolve variability across random directions. Do not relabel the fresh confirmation failure as success based on training or raw-energy improvements.

Next perform matched oracle capacity comparisons for the16, PCA20 and random20 output spaces on this same panel. Use per-case Fisher-optimal coefficients and evaluate finite KL, with true raw projections as a second metric. If PCA20 improves oracle behavior substantially, its learned failure points toward coefficient estimation/selection; if its oracle capacity barely improves, the training-energy basis is a weak way to find behaviorally important missing directions. These are post-hoc diagnostics, not additional fresh tests, and Fisher optima are not certified finite-KL ceilings. Preserve the overall objective and the original metric hierarchy rather than optimizing only raw reconstruction.
