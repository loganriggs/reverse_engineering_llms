# Destination-conditioned Fisher geometry of operator replacements

Fifty-three exact norm-and-angle-matched replacements for L4H1 are propagated at scale 0.05. Their logit tangents define a Fisher Gram matrix under the native next-token distribution. This quotient asks which operator differences the fixed router, destination state, and nonlinear suffix can see.

## Locked hypotheses

- **F1_destination_fisher_geometry_is_low_rank: not supported.**
- **F2_fisher_cosine_geometry_is_split_stable: supported.**
- **F3_leading_fisher_subspace_is_split_stable: supported.**
- **F4_raw_operator_geometry_predicts_destination_geometry: supported.**
- **F5_fisher_diagonal_predicts_small_finite_kl: supported.**

Fisher rank90 A/B: `27/27`; participation ratio `3.2/3.1`.
Cosine-geometry split correlation: `0.941`. Leading-8 subspace mean squared canonical cosine: `0.877`.
Raw-operator/Fisher cosine Spearman A/B: `0.313/0.321`.
Fisher diagonal/finite-KL Spearman A/B: `1.000/1.000`.

The designated L5H8 replacement has absolute loading `0.126/0.123` on the dominant sensitive mode (rank `49/50` among 53 candidates).

Largest discovery-A sensitive-mode loadings:

- L1H1: `+0.189`
- L0H0: `+0.155`
- L2H3: `+0.152`
- L0H8: `+0.150`
- L1H8: `+0.149`
- L2H5: `+0.148`
- L0H3: `+0.148`
- L0H4: `+0.146`
- L1H6: `+0.146`
- L4H3: `+0.146`
- L3H8: `+0.145`
- L2H2: `+0.143`

## Guardrails

This geometry is destination- and corpus-conditioned. It is an ontology of substitutability for one target function, not an intrinsic semantic dictionary. The Fisher map is estimated by a small finite scale rather than an analytic JVP, although quadratic scaling was independently checked. Low rank would justify a compact compatibility basis; high rank would mean stable substitutability still requires many destination-specific coordinates.
