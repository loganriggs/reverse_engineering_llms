

## H373 — Probability-defined output axes improve some finite responses but do not yield a sufficient compact subspace

Construct one global discovery metric K=Wᵀdiag(mean p)W−mean(μ μᵀ), where μ=E_p[W]. Average over the four positions and128 documents stored by H370. Its descending eigenvectors depend only on earlier discovery probabilities and native weights, not H372 response outcomes. Compare this mean-within probability basis with H364 weight-only covariance eigenvectors at ranks2,16,64,256. Freeze ladder and gates before finite outcomes. Reuse the H372512-document panel and native responses; this is a new diagnostic on a previously analyzed panel, not another independent confirmation.

Project the finite post-RMS activation delta Δ Euclideanly onto each retained subspace, add it to the intact normalized activation, and apply unembedding plus the native logit cap without another normalization. Recovery=1−KL(native target || patch)/KL(native target || unchanged baseline). Negative recovery means worse than leaving the baseline unchanged. Also compute the local Fisher error under the recipient’s actual probability/cap derivative. No response coefficients are fitted.

|Head|Family|Probability rank2|Probability rank16|Probability rank64|Probability rank256|Weight rank256|
|---|---|---:|---:|---:|---:|---:|
|0|actual|0.399|0.477|0.643|0.772|0.526|
|0|moment|0.111|0.248|0.404|0.599|0.160|
|0|donor|0.115|0.252|0.411|0.607|0.183|
|1|actual|0.265|0.338|0.458|0.624|-0.069|
|1|moment|0.173|0.259|0.407|0.601|-0.132|
|1|donor|0.183|0.266|0.414|0.603|-0.141|
|6|actual|0.183|0.196|0.307|0.526|0.198|
|6|moment|-0.081|-0.245|-0.084|0.185|0.241|
|6|donor|-0.054|-0.220|-0.072|0.182|0.260|

All frozen gates fail: {'probability64_sufficient': False, 'probability256_sufficient': False, 'probability16_advantage': False}. Neither probability rank64 nor rank256 achieves universal80% lower-bound recovery. Rank256 point recovery spans18.2–77.2%. The probability-rank16 advantage over weight rank16 is positive with joint lower bounds above0.1 for six of nine groups, but not universally: head0 actual has a lower bound0.007, and both head6 controls include zero. Keep all failures; no rank chosen post hoc as a successful compact representation.

All72 recovery estimates have simultaneous bands of approximately±0.192. Probability16-minus-weight16 has separate joint9 bands. The figure shows point estimates; intervals are saved in JSON and reported in the gate calculations. Bootstrap4,000 recipient-document/control-draw resamples, conditional on fixed discovery geometry and empirical donor assignments.

For heads0/1, finite and local recoveries are close. For head6 controls, probability rank256 local recovery is39.9–40.3%, falling to18.2–18.5% at full amplitude. Thus changing the basis does not remove the previously observed finite nonlinear effect. Poor Euclidean projection recovery is not proof that no coefficients within the subspace could approximate the response: the projection objective does not optimize the recipient’s probability-metric error.

Audits: {'eigen_relative_residual': 4.186982261643716e-15, 'orthogonality_max': 3.661027713659958e-15, 'minimum_eigenvalue': 3.007679248608042e-05, 'full_target_kl': 0.0}. Probability covariance is positive definite here, eigen-equation and orthogonality errors are near machine precision, all finite KL values pass positivity/finite checks, and the full-target positive control gives exactly zero KL. Baseline KL spans0.000110–0.00518 nats. Frozen source hashes verified. Save full learned-from-discovery metric/eigenbasis, site-level finite and local errors, bootstrap arrays, script/config. Native weights unchanged.

Hypothesis update: a behavior-relevant shared metric improves on raw weight geometry for several responses, but changing axes alone does not produce self-contained low-dimensional behavioral units. This does not by itself prove native computation is high dimensional: context-dependent coefficient choice and finite curvature remain alternatives. We still need full native response vectors to conduct this diagnostic.

Next discriminating experiment: keep the same rank16/64 subspaces but choose coefficients by an oracle local Fisher least-squares fit at each recipient. Compare that with Euclidean projection, then evaluate both finite and local recovery. A large oracle improvement would implicate coefficient/metric mismatch rather than missing span; poor oracle recovery would implicate the span itself. Label this explicitly as a response-dependent oracle, not a compressed executable model or discovered semantic feature. Broader goal remains active.
