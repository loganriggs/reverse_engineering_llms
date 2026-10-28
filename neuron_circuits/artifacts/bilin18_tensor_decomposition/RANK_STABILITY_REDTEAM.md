# Downstream rank stability red-team

## Hypotheses

- **H1 within-corpus stability: not supported.**
- **H2 cross-corpus stability: not supported.**

| Comparison to FineWeb A | Score Spearman | Score cosine | Jaccard@128 | Jaccard@512 | Jaccard@1024 |
|---|---:|---:|---:|---:|---:|
| fineweb_b | 0.041 | 0.591 | 0.076 | 0.178 | 0.326 |
| wikitext | 0.042 | 0.349 | 0.099 | 0.189 | 0.337 |
| tinystories | 0.047 | 0.622 | 0.080 | 0.180 | 0.335 |

A stable aggregate rank curve does not imply stable individual terms. These comparisons hold the inherited CP basis fixed and vary only the examples used to estimate downstream sensitivity.

Raw results: `rank_stability_redteam.json`.
