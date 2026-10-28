# Destination-conditioned Fisher geometry of operator replacements

Fifty-three exact norm-and-angle-matched replacements for L2H5 are propagated at scale 0.05. Their logit tangents define a Fisher Gram matrix under the native next-token distribution. This quotient asks which operator differences the fixed router, destination state, and nonlinear suffix can see.

## Locked hypotheses

- **F1_destination_fisher_geometry_is_low_rank: supported.**
- **F2_fisher_cosine_geometry_is_split_stable: supported.**
- **F3_leading_fisher_subspace_is_split_stable: supported.**
- **F4_raw_operator_geometry_predicts_destination_geometry: not supported.**
- **F5_fisher_diagonal_predicts_small_finite_kl: supported.**

Fisher rank90 A/B: `5/9`; participation ratio `1.4/1.5`.
Cosine-geometry split correlation: `0.924`. Leading-8 subspace mean squared canonical cosine: `0.712`.
Raw-operator/Fisher cosine Spearman A/B: `0.167/0.206`.
Fisher diagonal/finite-KL Spearman A/B: `1.000/1.000`.

The designated L3H8 replacement has absolute loading `0.134/0.131` on the dominant sensitive mode (rank `35/43` among 53 candidates).

Largest discovery-A sensitive-mode loadings:

- L1H4: `+0.208`
- L2H4: `+0.154`
- L0H3: `+0.150`
- L0H7: `+0.148`
- L2H8: `+0.146`
- L0H4: `+0.146`
- L2H7: `+0.145`
- L2H0: `+0.144`
- L0H6: `+0.143`
- L5H2: `+0.143`
- L3H7: `+0.142`
- L1H8: `+0.142`

## Guardrails

This geometry is destination- and corpus-conditioned. It is an ontology of substitutability for one target function, not an intrinsic semantic dictionary. The Fisher map is estimated by a small finite scale rather than an analytic JVP, although quadratic scaling was independently checked. Low rank would justify a compact compatibility basis; high rank would mean stable substitutability still requires many destination-specific coordinates.
