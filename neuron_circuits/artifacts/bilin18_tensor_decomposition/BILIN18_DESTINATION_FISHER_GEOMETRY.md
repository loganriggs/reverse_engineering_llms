# Destination-conditioned Fisher geometry of operator replacements

Fifty-three exact norm-and-angle-matched replacements for L3H8 are propagated at scale 0.05. Their logit tangents define a Fisher Gram matrix under the native next-token distribution. This quotient asks which operator differences the fixed router, destination state, and nonlinear suffix can see.

## Locked hypotheses

- **F1_destination_fisher_geometry_is_low_rank: not supported.**
- **F2_fisher_cosine_geometry_is_split_stable: supported.**
- **F3_leading_fisher_subspace_is_split_stable: supported.**
- **F4_raw_operator_geometry_predicts_destination_geometry: not supported.**
- **F5_fisher_diagonal_predicts_small_finite_kl: supported.**

Fisher rank90 A/B: `13/18`; participation ratio `1.9/2.3`.
Cosine-geometry split correlation: `0.944`. Leading-8 subspace mean squared canonical cosine: `0.839`.
Raw-operator/Fisher cosine Spearman A/B: `0.231/0.246`.
Fisher diagonal/finite-KL Spearman A/B: `1.000/0.999`.

The compatible L2H5 replacement has absolute loading `0.057/0.046` on the dominant sensitive mode (rank `53/53` among 53 candidates).

Largest discovery-A sensitive-mode loadings:

- L3H5: `+0.177`
- L3H3: `+0.161`
- L2H8: `+0.147`
- L1H4: `+0.146`
- L3H4: `+0.145`
- L5H2: `+0.145`
- L3H7: `+0.144`
- L2H6: `+0.143`
- L0H3: `+0.143`
- L2H2: `+0.143`
- L5H0: `+0.143`
- L2H7: `+0.142`

## Guardrails

This geometry is destination- and corpus-conditioned. It is an ontology of substitutability for one target function, not an intrinsic semantic dictionary. The Fisher map is estimated by a small finite scale rather than an analytic JVP, although quadratic scaling was independently checked. Low rank would justify a compact compatibility basis; high rank would mean stable substitutability still requires many destination-specific coordinates.
