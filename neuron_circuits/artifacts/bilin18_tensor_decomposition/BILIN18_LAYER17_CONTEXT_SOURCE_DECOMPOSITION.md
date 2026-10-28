# Architectural sources of the Layer-17 context register

The exact pre-normalization context state is split into carried state, direct original-embedding reinjection, and final attention. Position-conditioned mean filling preserves a common baseline before the native RMS normalization and tensor contraction.

## Locked hypotheses

- **P0_three_sources_recompose_native_register: supported.** All three exact pre-normalization sources reproduce the native context operator to relative error at most 1e-6.
- **P1_embedding_variation_is_a_strong_router: not supported.** Direct Layer-17 embedding-reinjection variation alone retains at least 0.75 operator fidelity on every corpus.
- **P2_embedding_and_carried_state_are_nearly_sufficient: supported.** Embedding plus carried-state variation retains at least 0.95 operator fidelity on every corpus.
- **P3_attention_is_a_small_operator_correction: supported.** The absolute three-source subset-Shapley allocation of final attention is at most 0.10 on every corpus.
- **P4_embedding_is_the_largest_shapley_source: not supported.** Embedding has the largest three-source subset-Shapley allocation on every corpus.
- **P5_attention_alone_is_not_a_router: not supported.** Attention variation alone retains at most 0.25 operator fidelity on every corpus.
- **P6_carried_state_adds_beyond_embedding: supported.** Adding carried-state variation to embedding improves fidelity by at least 0.05 on every corpus.
- **P7_embedding_dominance_transfers_ood: supported.** Embedding-only fidelity differs by at most 0.10 across FineWeb, WikiText, and TinyStories.

Layer-17 carried/embedding coefficients: `1.0234` / `8.0000`. Full-source coefficient error: `4.85e-07`.

## Source-subset operator fidelity

| Corpus | mean | carried | embedding | attention | carried+embedding | embedding+attention | carried+attention | full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb | 0.581 | 0.990 | 0.582 | 0.654 | 0.990 | 0.655 | 1.000 | 1.000 |
| wikitext | 0.606 | 0.988 | 0.607 | 0.678 | 0.989 | 0.679 | 1.000 | 1.000 |
| tinystories | 0.652 | 0.995 | 0.653 | 0.721 | 0.995 | 0.721 | 1.000 | 1.000 |

## Finite subset-Shapley allocation and removal cost

| Corpus | Shapley carried / embedding / attention | Removal carried / embedding / attention |
|---|---:|---:|
| fineweb | 0.377/0.000/0.041 | 0.345/0.000/0.010 |
| wikitext | 0.352/0.000/0.042 | 0.321/0.000/0.011 |
| tinystories | 0.311/0.000/0.037 | 0.279/0.000/0.005 |

## Interpretation boundary

These are allocations of finite variation around position-conditioned discovery means. The carried source already contains computations and earlier embedding injections, so the rows are not exclusive storage percentages. They identify which source variation selects the final local operator at this boundary.
