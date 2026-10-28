# Reusable vocabulary of transported tangent states

A rank-64 residual basis is fit jointly to all three candidate-replacement bundles on discovery half A at four common absolute boundaries. It is evaluated on held-out half B against target-specific PCA and seeded random subspaces. The causal test projects the exact tangent at a boundary and then recomputes the complete analytic attention/MLP suffix before measuring output-Fisher response.

## Locked hypotheses

- **S1_shared_state_vocabulary_is_locally_compact: not supported.**
- **S2_shared_vocabulary_is_causally_sufficient: not supported.**
- **S3_shared_vocabulary_beats_random: not supported.**
- **S4_shared_basis_is_split_stable: not supported.**
- **S5_vocabulary_is_cross_target_not_just_pooled: supported.**
- **S6_state_compressibility_increases_with_depth: not supported.**
- **attention_jvp_forward_invariant: supported.**

## Held-out frontier

| Target | Boundary | Shared/target/random energy | Shared/target/random Fisher fidelity |
|---|---:|---:|---:|
| L0H0 | 6 | 0.337 / 0.334 / 0.056 | 0.402 / 0.365 / 0.085 |
| L0H0 | 10 | 0.237 / 0.214 / 0.057 | 0.411 / 0.301 / 0.106 |
| L0H0 | 14 | 0.178 / 0.160 / 0.057 | 0.346 / 0.210 / 0.103 |
| L0H0 | 17 | 0.578 / 0.563 / 0.047 | 0.306 / 0.158 / 0.168 |
| L2H2 | 6 | 0.259 / 0.260 / 0.056 | 0.125 / 0.194 / 0.087 |
| L2H2 | 10 | 0.207 / 0.197 / 0.056 | 0.380 / 0.303 / 0.126 |
| L2H2 | 14 | 0.172 / 0.164 / 0.056 | 0.328 / 0.264 / 0.110 |
| L2H2 | 17 | 0.630 / 0.625 / 0.047 | 0.100 / 0.052 / 0.025 |
| L5H3 | 6 | 0.418 / 0.421 / 0.056 | 0.318 / 0.304 / 0.073 |
| L5H3 | 10 | 0.216 / 0.208 / 0.057 | 0.430 / 0.401 / 0.119 |
| L5H3 | 14 | 0.176 / 0.170 / 0.055 | 0.423 / 0.389 / 0.095 |
| L5H3 | 17 | 0.596 / 0.592 / 0.047 | 0.295 / 0.297 / 0.046 |

Shared-basis split stability:

- Layer 6: mean squared canonical cosine `0.391`
- Layer 10: mean squared canonical cosine `0.321`
- Layer 14: mean squared canonical cosine `0.249`
- Layer 17: mean squared canonical cosine `0.325`

## Guardrails

This is a vocabulary for a particular family of operator-replacement tangents, not all native activations. PCA directions may rotate within a stable subspace and are not automatically semantic. Local energy and downstream Fisher fidelity are different objectives; success on one cannot substitute for the other. A shared basis can also reflect architecture-wide residual anisotropy rather than reusable human concepts, which requires matched native-state and semantic controls next.
