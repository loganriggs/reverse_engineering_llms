# Pair-synergy router for the Layer-8 suffix atlas

The router sees invariant singleton norms plus exact pair-response norms, second-order synergy norms, and signed synergy/additive cosines. No triple or full-suffix response is used as an input.

## Locked hypotheses

- **P0_pair_synergy_router_validates: not supported.**
- **P1_pair_synergy_adds_value: not supported.**
- **P2_pair_synergy_features_are_used: supported.**
- **P3_second_order_path_grammar_is_insufficient: supported.**

Selected ridge alpha `1000.0`.

| Split | Median fidelity | 10th percentile | Minimum |
|---|---:|---:|---:|
| fineweb_cv | 0.989 | 0.574 | -2.818 |
| wikitext | 0.976 | 0.510 | -2.622 |
| wikitext_shuffled | 0.973 | -1.379 | -4.159 |

A pass would provide a layer-level second-order path grammar. Failure means that pair synergies alone do not robustly choose triples; it does not rule out head-level decomposition or true third-order propagation.
