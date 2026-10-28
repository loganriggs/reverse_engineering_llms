# Layer-17 context-router red team

The compact context-conditioned operator is retained. This test asks whether its apparent current-token routing survives stronger lexical, frequency, position, and embedding controls.

## Locked hypotheses

- **R0_tensor_metric_matches_explicit_operator: supported.** The 16-dimensional tensor Gram metric reproduces explicit operator fidelity to error at most 1e-6.
- **R1_exact_token_beats_matched_identity_null: supported.** Exact current-token routing exceeds the p95 of 64 coarse-class- and frequency-matched identity permutations by at least 0.03 on every corpus.
- **R2_exact_token_adds_beyond_coarse_class: supported.** Exact token identity explains at least 5% of the operator error remaining after the coarse-class router on every corpus.
- **R3_repeated_tokens_strengthen_identity_routing: not supported.** Among tokens observed at least four times in discovery, exact-token routing exceeds the matched coarse-class router by at least 0.05 fidelity on every corpus.
- **R4_token_routing_transfers_across_position_halves: supported.** The mean early-to-late and late-to-early exact-token router reaches at least 0.75 operator fidelity with at least 0.40 coverage on every corpus.
- **R5_embedding_geometry_routes_unseen_contexts: supported.** A discovery-frozen ridge map from the native current-token embedding reaches at least 0.75 operator fidelity on every corpus.
- **R6_position_alone_is_not_the_router: supported.** Position-only routing remains at least 0.05 below exact-token routing on every corpus.
- **R7_coarse_position_does_not_explain_exact_identity: not supported.** Coarse-class-plus-position routing remains at least 0.03 below exact-token routing on every corpus.
- **R8_token_position_lookup_is_not_required: supported.** Exact-token routing is within 0.05 of the more specific token-plus-position lookup on every corpus.

Tensor-Gram versus explicit-operator fidelity error: `2.92e-09`.

## Router comparison

| Corpus | mean / position | coarse / coarse+position | exact token / token+position | embedding ridge | matched null p95 | token residual gain | cross-position fidelity/coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| fineweb | 0.711/0.711 | 0.790/0.787 | 0.846/0.808 | 0.881 | 0.778 | 0.265 | 0.835/0.604 |
| wikitext | 0.721/0.721 | 0.798/0.795 | 0.808/0.765 | 0.850 | 0.764 | 0.052 | 0.796/0.476 |
| tinystories | 0.736/0.736 | 0.809/0.806 | 0.893/0.846 | 0.895 | 0.827 | 0.439 | 0.882/0.654 |

## Training-count conditional fidelity

Each cell is `examples: coarse / exact-token / embedding-ridge` in the tensor-induced operator metric.

### fineweb

| Minimum discovery count | Examples | Coarse | Exact token | Embedding ridge |
|---:|---:|---:|---:|---:|
| 1 | 21556 | 0.796 | 0.896 | 0.903 |
| 2 | 18415 | 0.794 | 0.907 | 0.910 |
| 4 | 15720 | 0.791 | 0.915 | 0.915 |
| 8 | 13834 | 0.784 | 0.917 | 0.917 |
| 16 | 11777 | 0.779 | 0.915 | 0.915 |

### wikitext

| Minimum discovery count | Examples | Coarse | Exact token | Embedding ridge |
|---:|---:|---:|---:|---:|
| 1 | 7303 | 0.833 | 0.868 | 0.870 |
| 2 | 5781 | 0.837 | 0.886 | 0.888 |
| 4 | 4580 | 0.853 | 0.897 | 0.897 |
| 8 | 3935 | 0.860 | 0.912 | 0.913 |
| 16 | 3264 | 0.872 | 0.907 | 0.907 |

### tinystories

| Minimum discovery count | Examples | Coarse | Exact token | Embedding ridge |
|---:|---:|---:|---:|---:|
| 1 | 9443 | 0.815 | 0.920 | 0.921 |
| 2 | 7918 | 0.809 | 0.931 | 0.931 |
| 4 | 6982 | 0.804 | 0.933 | 0.933 |
| 8 | 5936 | 0.793 | 0.937 | 0.937 |
| 16 | 4847 | 0.787 | 0.933 | 0.933 |

## Interpretation boundary

Exact-token success would identify a reusable lexical prior over local operators, not a complete context-independent instruction: the remaining error still depends on the full hidden state. Embedding-ridge success would show that this prior extends through learned lexical geometry rather than only repeated token IDs. The tensor core and context subspace remain basis-invariant even if every lexical router fails.
