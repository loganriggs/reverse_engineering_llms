# Native-state control for the tangent vocabulary

The failed rank-64 tangent vocabulary is compared with an uncentered native residual-state PCA using the same rows, absolute boundaries, rank, and projection operation. The native basis is fit on half A. Held-out tangent energy and full-suffix Fisher fidelity are evaluated on half B.

## Locked hypotheses

- **N1_tangent_basis_has_specific_local_advantage: not supported.**
- **N2_tangent_basis_has_specific_causal_advantage: not supported.**
- **N3_tangent_and_native_subspaces_are_distinct: supported.**
- **N4_native_state_basis_is_more_split_stable: not supported.**
- **N5_static_vocabulary_failure_is_not_just_native_geometry: not supported.**
- **attention_jvp_forward_invariant: supported.**

## Matched comparison

| Target | Boundary | Tangent/native energy | Tangent/native Fisher fidelity |
|---|---:|---:|---:|
| L0H0 | 6 | 0.337 / 0.320 | 0.402 / 0.367 |
| L0H0 | 10 | 0.237 / 0.209 | 0.411 / 0.362 |
| L0H0 | 14 | 0.178 / 0.164 | 0.346 / 0.366 |
| L0H0 | 17 | 0.578 / 0.569 | 0.306 / 0.335 |
| L2H2 | 6 | 0.259 / 0.236 | 0.125 / 0.217 |
| L2H2 | 10 | 0.207 / 0.179 | 0.380 / 0.374 |
| L2H2 | 14 | 0.172 / 0.157 | 0.328 / 0.360 |
| L2H2 | 17 | 0.630 / 0.621 | 0.100 / 0.201 |
| L5H3 | 6 | 0.418 / 0.342 | 0.318 / 0.305 |
| L5H3 | 10 | 0.216 / 0.173 | 0.430 / 0.472 |
| L5H3 | 14 | 0.176 / 0.152 | 0.423 / 0.434 |
| L5H3 | 17 | 0.596 / 0.584 | 0.295 / 0.328 |

## Subspace controls

| Boundary | Native split stability | Tangent split stability | Tangent/native overlap |
|---:|---:|---:|---:|
| 6 | 0.346 | 0.391 | 0.320 |
| 10 | 0.306 | 0.321 | 0.339 |
| 14 | 0.270 | 0.249 | 0.373 |
| 17 | 0.331 | 0.325 | 0.446 |

## Guardrails

Native PCA and tangent PCA optimize different data distributions; neither is behaviorally supervised. Uncentered native covariance intentionally tests the strongest ordinary residual anisotropy control, including mean directions. If both bases fail causally, that favors a context-conditioned or Fisher-weighted chart, not the conclusion that no compact causal coordinates exist.
