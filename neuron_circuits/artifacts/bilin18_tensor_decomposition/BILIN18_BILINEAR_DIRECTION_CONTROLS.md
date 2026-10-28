## H786 — Random directions reproduce the L1/L4 nonlinear-error ordering

Previous turn was progress: H785 established an L1/L4 difference under matched angular input perturbations across eight task families, with direct-input versus analytic-output verification. H786 tests whether that difference requires task-related directions.

At exactly the same H785 base states, compare the true donor displacement, the next sorted item's displacement within the same family and edit type, and three seeded isotropic Gaussian directions. All directions are projected into the current base state's tangent plane, normalized, then rotated on the same constant-norm sphere at chord/input ratios 0.05, 0.15 and 0.30. Shuffling never pairs an item with itself. The random seed is deterministic per item, edit and control, shared across layers before projection. Geometry is measured at all radii; full versus first-order causal output interventions are evaluated at radius 0.30 through all downstream model paths.

The prediction and control definitions were saved before model evaluation. This is a controlled follow-up on the reused H785 panel, not new lexical data. Random directions are size-matched in normalized input space; they do not match activation covariance, module-output magnitude, or downstream gain.

### Result that changes the research interpretation

L1's module-output first-order approximation error exceeds L4's for all eight families, all three radii, both task/nuisance base-edit sets, and each of the three random seeds. The same ordering holds for shuffled directions. Thus the broad L1/L4 ordering does not require true task directions. It supports general layer geometry as an explanation and weakens the interpretation that the ordering itself identifies task-specific features.

This is not evidence that every detail is generic, or that all task directions are equivalent to random. At radius 0.30, competence-passing families have the following pooled relative module errors (interaction norm divided by full-change norm, aggregated in squared norm):

| Family | L1 true | L1 shuffled | L1 random pooled | L4 true | L4 shuffled | L4 random pooled |
|---|---:|---:|---:|---:|---:|---:|
| Agreement | .641 | .631 | .551 | .303 | .304 | .276 |
| Capital recall | .584 | .541 | .539 | .322 | .305 | .274 |
| Entity binding | .641 | .622 | .618 | .262 | .261 | .246 |
| Copying | .593 | .551 | .533 | .263 | .259 | .266 |

Random-pooled combines three draws per base state; these are not independent task observations. The within-family shuffled control retains task-family structure and often nearly reproduces true-direction geometry. It is not a null of “no task information.”

### Geometry and behavioral specificity remain different

At L1 and radius 0.30, the true/random first-order answer-margin error RMS is .250/.080 for agreement, .143/.168 for capital recall, .145/.261 for entity binding, and .133/.254 for copying. True directions do not uniformly produce greater behavioral approximation error. These are absolute discrepancies, not causal attribution fractions.

Donor-directed mean margin change is also mixed. For L1 entity binding it is +.209 for true directions, +.198 for shuffled directions and +.163 for random directions. For L1 copying it is -.246, -.032 and -.119 respectively. A positive value denotes movement toward the donor's intended answer contrast; these spherical final-position perturbations need not move the answer as a native donor swap would. Random controls can cause structured, base-dependent effects, so positive mean movement alone would not verify a reusable semantic direction.

The same competence restrictions apply: agreement, capital recall, entity binding and copying pass; the other four families fail the predeclared gate and support only geometric descriptions, not successful-task claims. Reciprocal edits, items, families, and seeds are correlated; sentiment/negation nuisance native contexts are duplicated.

### Verification

5,760 module-geometry cases and 4,096 whole-model forwards, including 256 zero-intervention controls. Constant norm preserved to 1.12e-15 relative error and chord radius to 1.67e-16 absolute error. Recomputed true-direction patches replay H785 within 4.77e-7 maximum component error; true-direction answer effects replay within 4.77e-6. Identity logit discrepancies across batches are <=1.72e-5. All effects finite; input, reference, runner and model hashes unchanged. Original reserved test untouched; natural-document pool remains 26,112. No training or featurizer selection was performed.

### Next high-information discriminator

A constant-norm rotation has a sign-independent radial component: for opposite tangent directions, v_plus=cos(theta)u+sin(theta)||u||t and v_minus=cos(theta)u-sin(theta)||u||t. Both include (cos(theta)-1)u. The quadratic map can therefore produce a common even response, including for random directions. H786 does not distinguish that common response from direction-specific odd responses.

Next evaluate paired positive/negative tangent rotations and separate their odd and even module changes and behavioral responses. Compare true, shuffled and random directions, with base-paired inference and competence-passing tasks emphasized. This can distinguish directional semantic effects from generic radial/quadratic responses before fitting more feature groups. Then refresh the broader ~20-hypothesis tournament and choose a transfer experiment targeting the best-supported alternatives. No meaningful manifold or fundamental computation unit is yet verified by these controls.
