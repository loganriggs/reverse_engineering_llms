# Data-matched red-team of semantic-family routes

Every same-family route is compared with sixteen outside-family routes learned from the same number of discovery examples and evaluated under the same finite patch.

## Locked hypotheses

- **M0_family_advantage_survives_data_matching: not supported.**
- **M1_family_route_wins_most_pairs: not supported.**
- **M2_family_route_is_head_specific: supported.**
- **M3_advantage_is_not_one_family: supported.**

| Corpus | Family effect | Matched outside | Advantage | Pair win rate | Family head-shift |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.211 | 0.165 | 0.024 | 0.684 | 0.056 |
| tinystories | 0.139 | 0.126 | 0.016 | 0.895 | 0.058 |

| Family | FineWeb advantage | TinyStories advantage |
|---|---:|---:|
| sentiment | 0.005 | 0.047 |
| gender | 0.018 | 0.021 |
| tense | 0.046 | 0.006 |
| increasing_scale | -0.022 | 0.019 |
| polarity | 0.038 | 0.008 |

A positive matched advantage is preliminary family-level route reuse. With only three or four lexical pairs per family, it remains vulnerable to token-frequency and syntactic-position confounds and requires new pairs for confirmation.
