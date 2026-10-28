# Structure of per-site bilinear channel coalitions

Each site contributes its 32 largest exact final-block channel effects. Overlap is intersection divided by 32. The entropy-effective bank size measures how broadly selection frequency is distributed across all 192 channels. Relation and answer comparisons use every held-out pair of sites.

## Locked hypotheses

- **S0_mixed_coalitions_overlap_less: supported.**
- **S1_mixed_uses_a_broader_effective_bank: supported.**
- **S2_pure_coalitions_are_relation_organized: not supported.**
- **S3_mixed_relation_organization_is_weak: supported.**
- **S4_fixed_bank_tracks_pure_site_oracles: not supported.**

| Seed | rho | Mean overlap | Effective bank | Fixed-bank capture | Relation advantage | Answer advantage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.707 | 49.5 | 0.761 | 0.007 | 0.072 |
| 0 | 0.5 | 0.269 | 138.1 | 0.386 | 0.001 | 0.003 |
| 1 | 0.0 | 0.505 | 73.6 | 0.627 | 0.050 | 0.047 |
| 1 | 0.5 | 0.259 | 138.9 | 0.365 | 0.004 | 0.013 |

Selection is based on effect norm and is not an optimal subset under cancellation. Pairwise sites within one trained model are correlated; the two seeds, not site pairs, are the replication units.
