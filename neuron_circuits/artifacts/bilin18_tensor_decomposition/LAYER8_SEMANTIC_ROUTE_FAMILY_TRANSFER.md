# Semantic-family versus generic attention-route transfer

Layer-4 edge routes learned from held-out lexical pairs are compared with outside-family, global, pair-specific, oracle, and cyclic-head controls under exact finite counterfactual patching.

## Locked hypotheses

- **F0_semantic_family_route_transfers: not supported.**
- **F1_generic_broadcast_route_explains_transfer: not supported.**
- **F2_lexical_pair_route_is_required: not supported.**
- **F3_cross_pair_routes_are_causal: not supported.**
- **F4_oracle_gap_remains: supported.**

| Corpus | Family | Outside family | Global | Pair-specific | Oracle | Family shift | Global shift |
|---|---:|---:|---:|---:|---:|---:|---:|
| fineweb | 0.211 | 0.157 | 0.183 | 0.160 | 0.294 | 0.056 | 0.122 |
| tinystories | 0.139 | 0.104 | 0.123 | 0.126 | 0.241 | 0.058 | 0.089 |

Values are median signed target projections for removal. A generic route can be mechanistically reusable without encoding the shared human semantic category; value content and the current state may supply the object being transported.
