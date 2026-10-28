## H694 — Search distillation improves group labels but not the reconstruction objective

Use H693's training-only exact search decisions at the original0.01group cost as binary targets. Fit the same linear and quadratic-energy gate families as H692 with binary cross entropy, matching initialization seeds,3000steps, learning rate and minibatch streams. Select checkpoints using the same independent validation reconstruction-plus-count objective. Teacher group labels do not enter training or checkpoint selection. Higher combinations are evaluated only after selection. This is conditional search distillation; the frozen dictionary was originally learned with supplied gates.

| Seed | Detector | Selected step | Validation group recovery | Validation objective | Held group recovery | Held relative MSE |
|---|---|---:|---:|---:|---:|---:|
|21|linear|0|4.74%|0.27061|8.94%|0.26877|
|21|energy|3000|86.87%|0.05103|73.44%|0.01826|
|22|linear|0|5.57%|0.33443|8.50%|0.32571|
|22|energy|2900|86.82%|0.05127|74.22%|0.01735|
|23|linear|0|6.01%|0.38895|8.01%|0.33606|
|23|energy|3000|87.01%|0.04954|73.44%|0.01764|

Energy gates now recover86.82–87.01% of validation teacher group patterns, versus1.42–11.38% under H692 straight-through reconstruction training. Agreement with exact search is87.35–87.60% on validation and78.52–79.20% on held combinations. Held teacher group recovery is73.44–74.22%. Therefore this same detector family can represent substantially more meaningful group decisions than direct reconstruction training found. This does not prove that it can implement the exact inference rule or that optimization is the only remaining limitation.

Crucially, validation objective worsens to0.04954–0.05127, versus0.03545–0.03695 under H692 energy gates; exact search achieves0.01600. Validation relative reconstruction error is3.34–3.52%, despite better labels. Classification accuracy does not weight a mistake by the reconstruction energy it removes. A cost-sensitive objective is needed before calling distillation a better featurizer. Held reconstruction errors1.74–1.83% also remain far above exact search0.0537%. The stronger labels and weaker reconstruction are both retained; neither metric alone settles quality.

Linear detector checkpoints all select step0: binary-classification training does not improve their validation reconstruction objective. A linear gate is poorly suited to presence that is symmetric under x→−x, but this finite experiment alone does not prove a general impossibility theorem. Energy detectors are sign-symmetric by construction and have644parameters versus68for linear; this is not an equal-budget superiority claim.

Next test cost-sensitive gate training using the full training reconstruction table: compute the exact expected reconstruction-plus-count cost under independent Bernoulli gate probabilities, summing all16patterns. That provides a differentiable, consequence-weighted objective without straight-through gradients or teacher labels. Compare hard decisions at evaluation; expected stochastic cost need not equal hard-threshold cost. This isolates another optimization choice before moving to joint learning, hierarchy/shared computation and the early-layer manifold/causal program.

Audits: all source hashes unchanged, finite results, training targets computed from training-only exact-search tables, no original-model forwards/new documents, reserved LLM test split untouched. Broad goal active.
