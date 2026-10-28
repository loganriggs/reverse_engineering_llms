# Prospective semantic-route transfer to new lexical pairs

Routes learned only from the original pairs are evaluated on seventeen newly declared substitutions, three contexts per pair, on FineWeb and TinyStories.

## Locked hypotheses

- **P0_correct_semantic_route_transfers_to_new_pairs: not supported.**
- **P1_correct_family_wins_most_new_examples: not supported.**
- **P2_correct_route_is_causal_and_head_specific: supported.**
- **P3_generic_broadcast_is_sufficient: supported.**
- **P4_oracle_gap_persists: not supported.**

| Corpus | Correct family | Wrong-family median | Advantage | Win rate | Global | Head shift | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| fineweb | 0.127 | 0.116 | 0.014 | 0.608 | 0.133 | 0.064 | 0.181 |
| tinystories | 0.175 | 0.156 | 0.016 | 0.706 | 0.188 | 0.102 | 0.236 |

| Family | FineWeb advantage/win | TinyStories advantage/win |
|---|---:|---:|
| sentiment | 0.008/0.56 | 0.008/0.67 |
| gender | 0.026/0.75 | 0.034/0.83 |
| tense | 0.004/0.56 | -0.008/0.44 |
| increasing_scale | -0.001/0.44 | 0.010/1.00 |
| polarity | 0.015/0.67 | 0.011/0.58 |

The semantic labels and all new pairs were frozen before evaluation. Success identifies reusable routing associated with the declared family; it still does not imply that an edge or head contains the concept independently of its value/state content.
