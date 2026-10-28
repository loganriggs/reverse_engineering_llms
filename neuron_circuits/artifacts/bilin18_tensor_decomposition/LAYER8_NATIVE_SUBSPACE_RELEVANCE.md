# Native relevance of canonical route subspaces

Stable three-layer route projectors are compared with the complete suffix-Jacobian projector and with seeded random rank-16 subspaces on the frozen blind examples.

## Locked hypotheses

- **N0_route_subspace_matches_full_native_subspace: supported.**
- **N1_route_subspace_matches_full_finite_effect: not supported.**
- **N2_full_native_subspace_beats_random: supported.**
- **N3_route_subspace_beats_random: supported.**
- **N4_stable_route_object_is_native_relevant: not supported.**

| Group | Route/full projector overlap median/q10 | Min principal cosine median | Route/full finite fidelity median/q10 | Route KL | Full KL | Random KL | Route/random | Full/random |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| failure | 0.942/0.914 | 0.791 | 0.868/0.471 | 5.435e-03 | 7.126e-03 | 3.314e-04 | 16.5x | 21.4x |
| control | 0.951/0.838 | 0.800 | 0.845/0.712 | 4.939e-03 | 7.126e-03 | 3.219e-04 | 18.5x | 21.8x |

The full projector is a local causal-sensitivity object, not a proof of natural feature use. All finite edits are norm-matched; literal natural-component removal and semantic transfer remain separate tests.
