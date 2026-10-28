## H958 — Most downstream-metric gains survive consistent feature states

Question: are H957's teacher-informed coefficients merely independent corrections for each directed intervention, or can a single set of state coordinates reproduce much of the gain? A fixed linear decoder requires each intervention coefficient to be z(donor)-z(recipient), with antisymmetric reversals and zero sums around closed loops.

For each four-state factorial quartet, fit centered state potentials to all12 directed oracle coefficient vectors by least squares on the graph incidence matrix. Use separate MLP/joint targets and a shared target averaging both paths. Retain both original budgets, both seeds, unscaled potentials, and common-quartet rescaling that preserves consistency while respecting every edge norm budget. Evaluate every candidate on both downstream paths, including cross-path transfer. Frozen .1/top8 span; no weight training. Per-quartet potentials remain teacher-informed and fit to observed states, not predicted from activations.

Original budget1 oracle inconsistency is substantial. Median triangle-closure RMS relative to edge RMS is.51 prepositional/.41 relative clause for joint-path coefficients; reversal inconsistencies .51/.38. Projection removes median28.5%/22.0% of coefficient norm (ranges22–44%/15–29%). Exact teacher and Euclidean projection controls obey consistency to9.16e-7 relative, supporting that the oracle violations are not floating-point artifacts.

Despite this coefficient change, much of the live output-distribution improvement survives. Budget1 joint-path KLratios (relative to zero MLP change), ranges across seeds:

| Factorial cell | Independent joint-edge oracle | Consistent joint states | Consistent shared states |
|---|---|---|---|
| prepositional:both | 0.351–0.498 | 0.364–0.529 | 0.365–0.530 |
| prepositional:distractor | 0.295–0.403 | 0.305–0.415 | 0.304–0.414 |
| prepositional:subject | 0.393–0.548 | 0.434–0.644 | 0.433–0.643 |
| relative_clause:both | 0.387–0.400 | 0.415–0.424 | 0.415–0.424 |
| relative_clause:distractor | 0.469–0.490 | 0.496–0.512 | 0.495–0.512 |
| relative_clause:subject | 0.253–0.296 | 0.294–0.312 | 0.295–0.313 |

Shared consistency raises joint KLratio modestly, to.295–.643 across these cells, compared with.253–.548 for the independent-edge oracle and.688–.998 for original learned top-eight effects (H957). Most of the downstream-metric gain therefore does not require unconstrained per-edge freedom. Sharing state coordinates across MLP-only and joint paths adds little penalty beyond within-path consistency here.

Task-margin tradeoffs remain: prepositional subject-only error rises from46–51% for independent edges to60–92% for consistent shared states, despite preserving substantial KL improvement. Relative-clause subject-only error rises from4.6–7.4% to12–13%. Thus a consistent distribution-focused representation is not automatically a good explanation of the selected task margin.

Norm constraints are not the main cost: minimum common rescaling is.926 at budget1; budget2 needs no rescaling. Exact graph projection is verified both by pseudoinverse and the complete-graph closed form. Consistency closure after projection is3.43e-16 relative; solution disagreement9.10e-13 absolute; capped norm ratios<=1. Native identity2.20e-5; historical joint-margin replay7.29e-6; original edge-oracle margins agree with H957 within1.15e-5 and KL within3.86e-8. Independent summary aggregation<1e-12. All11,424 expected live forwards completed; hashes unchanged; no new teacher endpoint data, model training, or reserved-test access; natural document pool26,112.

Interpretation: on these finite observed states, a consistent eight-dimensional teacher-informed representation preserves much of the behavioral-equivalence advantage. This is stronger than independent intervention corrections, but still much weaker than a reusable learned featurizer. Each quartet has its own additive gauge; no encoder predicts unseen states, and the disconnected quartets do not establish global identifiability. The constraint is necessary for fixed additive decoders, not for every nonlinear/context-conditioned computation model. Uniform coefficient-space projection is not globally KL-optimal; losses here do not prove a lower bound for consistent representations.

Next: predict these coordinates from activations with strict lexical holdouts. Fit differences of a shared encoder so arbitrary per-quartet offsets cannot leak into targets. Compare raw normalized-state linear features against the frozen64 bilinear product features, and retain the original top-eight encoder and teacher-informed state oracle as controls. Hold out complete noun families across both templates, with a separate cross-template transfer check. Fix regularization without tuning on held-out downstream scores. Success would show some learnable transfer of the downstream-metric representation; failure would distinguish finite-data existence from reusable computation.
