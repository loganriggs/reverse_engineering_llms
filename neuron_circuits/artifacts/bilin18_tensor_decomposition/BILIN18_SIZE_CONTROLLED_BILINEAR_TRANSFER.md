## H785 — L1's greater finite nonlinearity survives angular-size control across eight task families

Previous turn was progress: H784 verified exact factor decomposition and found a large first-order causal-readout error for finite L1 copied-value edits. H785 tests the predeclared prediction that the L1/L4 difference persists after matching normalized-input displacement size and extending beyond copying.

We took all eight transfer items from each of the eight established task families and added the fixed prefix “Complete the following sentence:” plus newline to every context. This creates new contexts but reuses underlying items and vocabulary; it is not an independent lexical replication. The manifest, layer choices, radii, prediction and competence gate were saved before model evaluation. No examples or families were selected based on observed intervention results.

At each final-position normalized MLP input u, project the donor direction onto u's tangent plane. Rotate along that direction on the sphere with radius ||u||, at chord distances rho*||u|| for rho=0.05, 0.15, 0.30. This keeps input norm fixed and makes displacement size equal relative to input norm across L1 and L4. It does not hold the donor direction, local Jacobian gain, activation distribution, or semantic change fixed across layers. The perturbations are synthetic intermediate states, not native prompt interpolations.

For the resulting displacement du, compare the exact bilinear output change against its first-order approximation at u. Apply each output change to the receiver's final-position MLP output and run all downstream computation. This readout covers all downstream paths from that site, unlike H784's two-pass isolation through L5H5 source values. Consequently, effect sizes from the two experiments are not directly comparable.

### Main result

L1 has greater module-output first-order approximation error than L4 in every task family at every tested radius, for both task and same-answer nuisance directions. This includes the four competence-passing families. These are descriptive comparisons of pooled errors, not independent significance tests.

| Relative chord distance | L1 relative output error range across task families | L4 relative output error range |
|---|---:|---:|
| 0.05 | 0.095–0.131 | 0.043–0.056 |
| 0.15 | 0.281–0.379 | 0.130–0.169 |
| 0.30 | 0.523–0.675 | 0.262–0.340 |

Relative output error is sqrt(sum ||interaction||^2 / sum ||full change||^2). It is a first-order approximation error, not an explained causal fraction. The tested finite interaction is measured around the receiver expansion point; midpoint linearization of a quadratic map would give an exact finite difference. Thus these results concern fixed-background linear approximations, not proof that a quadratic function cannot be represented through a moving derivative.

For competence-passing families at radius 0.30:

| Family | L1 relative module error | L4 relative module error | L1 answer-margin error RMS | L4 answer-margin error RMS |
|---|---:|---:|---:|---:|
| Agreement | 0.641 | 0.303 | 0.2498 | 0.0222 |
| Capital recall | 0.584 | 0.322 | 0.1431 | 0.0579 |
| Entity binding | 0.641 | 0.262 | 0.1453 | 0.0507 |
| Sequence copying | 0.593 | 0.263 | 0.1330 | 0.0331 |

Answer-margin error is the RMS discrepancy between first-order and full interventions. Full intervention effect RMS values for those families are respectively L1: 0.3067, 0.1754, 0.3941, 0.6812; L4: 0.0659, 0.2460, 0.2140, 0.1512. Output approximation error and behavioral sensitivity are different quantities; the L1/L4 relation in one does not imply the same effect magnitude in the other. No causal relative-error claims are made where effects are small.

### Competence and scope

The gate requires >=4 token-valid pairs, >=75% correct paired choices at both endpoints, and >=50% pooled intended full-vocabulary top1. Agreement, capital recall, entity binding and copying pass. Context association, negation, relational order and sentiment fail. All eight remain reported, but failed families do not support claims of successful task computation. Relational order has 0% paired choice correctness; context association has 100% paired choice correctness but 0% intended full-vocabulary top1, illustrating why contrast accuracy alone is insufficient.

Task edits use both directions; nuisance edits use the unchanged-answer context; identity edits use zero displacement. Sentiment and negation reuse the same base/nuisance contexts, so their nuisance results are duplicates, not independent evidence. Family and reciprocal-edit observations are correlated.

### Audits

64 pairs, 192 native contexts, no tokenization exclusions and no degenerate tangent directions. Main experiment: 3,264 model forwards. Sphere norm is preserved to 1.12e-15 relative error; target chord radius to 1.67e-16 absolute error. Analytic output replay has <=1.03e-6 relative L2 error. Identity logit discrepancy is <=1.67e-5 across differently batched executions; it is not exact zero. All effects are finite and the manifest hash is unchanged.

An independent implementation check directly replaces the normalized MLP input rather than adding an analytically computed output change. Across all task/nuisance edits, both layers, all eight families at radius 0.30 (384 additional forwards), direct-input versus analytic-output interventions agree within 1.01e-5 answer margin and 1.35e-6 output relative L2. This validates the construction without relying solely on its algebraic self-check. Total forwards including this audit: 3,648. Original reserved test untouched; natural-document pool remains 26,112.

### Research implication and next discriminator

The L1/L4 difference is not explained solely by the original edits having different displacement sizes, and it extends beyond copying at this final-position site. This supports evaluating featurizers on finite transformations across backgrounds. It does not establish semantic specificity, a reusable manifold, or a unique fundamental unit of computation.

The next critical alternative is general layer geometry: L1 might have larger finite interaction for arbitrary tangent directions. Compare seeded random, shuffled-task and true donor directions at the same base states/radii, with output geometry and downstream sensitivity evaluated separately. If random controls reproduce the difference, prioritize context-dependent quadratic geometry over task-specific interaction features. If true directions show additional reproducible structure, test whether shared features or small computational graphs capture that excess. Refresh the broader hypothesis tournament after these controls rather than adding more unmotivated architecture variants.
