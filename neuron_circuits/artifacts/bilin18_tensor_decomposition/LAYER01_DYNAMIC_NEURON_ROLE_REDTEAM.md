# Red-team of dynamic bilinear-neuron factor roles

Pair-bootstrap enrichment controls for the possibility that bilinear products are generically one-sided. Side labels are shuffled across recurring channels while preserving occurrence counts. Cross-corpus side and sign labels are shuffled within each lexical-pair intersection, preserving marginal imbalance.

## Locked hypotheses

- **N0_asymmetry_is_enriched_by_causal_selection: supported.**
- **N1_recurring_side_roles_beat_shuffle: supported.**
- **N2_cross_corpus_side_roles_beat_shuffle: not supported.**
- **N3_cross_corpus_sign_stability_beats_shuffle: not supported.**

| Layer | Corpus | Asymmetry enrichment (95% pair CI) | Recurring side agreement / null median / p |
|---:|---|---:|---:|
| 0 | fineweb | 0.060 (0.044, 0.079) | 0.667 / 0.600 / 0.0115 |
| 0 | tinystories | 0.050 (0.034, 0.072) | 0.667 / 0.600 / 0.0035 |
| 1 | fineweb | 0.070 (0.052, 0.079) | 0.667 / 0.625 / 0.0125 |
| 1 | tinystories | 0.071 (0.053, 0.086) | 0.667 / 0.600 / 0.0185 |

| Layer | Cross-corpus side observed/null/p | Cross-corpus sign observed/null/p |
|---:|---:|---:|
| 0 | 0.732/0.512/0.0005 | 1.000/1.000/1.0000 |
| 1 | 0.522/0.500/0.2209 | 1.000/1.000/1.0000 |

The sign test is essential because 97% of contribution-selected coordinates are positive; raw agreement near one can therefore be a class-imbalance artifact. Factor-side names are arbitrary under a per-neuron swap, but agreement and its shuffle null are invariant to a fixed swap.
