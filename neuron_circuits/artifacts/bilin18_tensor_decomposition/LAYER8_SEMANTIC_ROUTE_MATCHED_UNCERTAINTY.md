# Uncertainty audit for data-matched semantic routes

This is a post-result uncertainty analysis; it does not change the locked verdicts or thresholds.

| Corpus | Median advantage (bootstrap 95%) | Positive pairs | One-sided sign p |
|---|---:|---:|---:|
| fineweb | 0.024 [-0.005, 0.046] | 13/19 | 0.0835 |
| tinystories | 0.016 [0.008, 0.046] | 17/19 | 0.0004 |

Positive on both corpora: `11/19`; cross-corpus Pearson/Spearman of pair advantages: `0.003/-0.258`.

TinyStories supplies evidence for a small positive family advantage, while FineWeb remains compatible with zero and pairwise advantage magnitudes do not replicate strongly. The correct label is weak/exploratory family modulation, not a validated semantic verb.
