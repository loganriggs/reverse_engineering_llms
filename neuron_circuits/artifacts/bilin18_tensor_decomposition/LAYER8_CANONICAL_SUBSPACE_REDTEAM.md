# Canonical sensitivity subspace red-team

The same blind failures and frozen controls are represented by rank-16 projectors of the unit-input Fisher sensitivity operator instead of discontinuous greedy neuron sets.

## Locked hypotheses

- **C0_pairwise_route_recovers_canonical_subspace: supported.**
- **C1_canonical_subspace_recovers_finite_effect: supported.**
- **C2_leading_direction_is_stable_when_identified: supported.**
- **C3_sparse_failures_are_extractor_specific: supported.**

| Group | Valid modes | Gram rel. error | Rank-16 overlap median/min | Minimum principal cosine median | Rank-boundary gap median | Subspace finite fidelity median/q10 | Leading cosine median | Leading finite fidelity median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| failure | 512 | 0.0268 | 1.000/0.997 | 0.998 | 1.082 | 0.999/0.992 | 1.000 | 1.000 |
| control | 512 | 0.0156 | 1.000/0.996 | 0.999 | 1.111 | 1.000/0.928 | 1.000 | 1.000 |

Sparse infinitesimal fidelity median: failures `-0.030`, controls `0.999`.

Projector overlap is the primary identification metric: individual eigenvectors are not canonical inside a nearly degenerate eigenspace. These are sensitivity subspaces for declared route Jacobians, not yet semantic modules or sufficient reconstructions of natural computation.
