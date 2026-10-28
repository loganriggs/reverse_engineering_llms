# Final-query MLP write subspaces in Layers 0 and 1

Unit-normalized final-position MLP write changes from the original nineteen substitutions and two FineWeb contexts define frozen family and global SVD bases. Ranks 1/2/4/6 are projected into seventeen new substitutions and propagated through the complete suffix. Fixed random bases and wrong-family bases are causal controls.

## Locked hypotheses

- **S0_compact_global_subspace_transfers: not supported.**
- **S1_family_subspace_is_specific: not supported.**
- **S2_family_subspace_beats_global: not supported.**
- **S3_learned_subspace_beats_random: not supported.**
- **S4_subspace_transfers_ood: supported.**
- **S5_rank4_is_near_rank6: supported.**

| Layer | Corpus | Rank-6 correct / wrong / advantage (win) | Global | Random | Local correct | Native-256 reference |
|---:|---|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.018 / 0.011 / -0.015 (0.45) | 0.027 | 0.010 | 0.015 | 0.154 |
| 0 | tinystories | 0.019 / 0.019 / 0.005 (0.51) | 0.053 | 0.011 | 0.009 | 0.180 |
| 1 | fineweb | 0.019 / 0.011 / 0.009 (0.53) | 0.041 | 0.007 | 0.011 | 0.065 |
| 1 | tinystories | 0.053 / 0.013 / 0.033 (0.61) | 0.031 | 0.000 | 0.012 | 0.064 |

| Layer | Corpus | Family rank frontier 1 / 2 / 4 / 6 | Global frontier |
|---:|---|---:|---:|
| 0 | fineweb | 0.004 / 0.005 / 0.009 / 0.018 | 0.002 / -0.001 / 0.006 / 0.027 |
| 0 | tinystories | -0.002 / 0.008 / 0.011 / 0.019 | -0.000 / -0.002 / 0.048 / 0.053 |
| 1 | fineweb | 0.001 / 0.003 / 0.008 / 0.019 | -0.001 / 0.014 / 0.040 / 0.041 |
| 1 | tinystories | 0.010 / 0.019 / 0.036 / 0.053 | 0.001 / 0.002 / 0.012 / 0.031 |

A compact global subspace would support a reusable query-position effect interface even when native family banks fail. A family-over-global advantage is the stronger semantic claim; without it, the subspace is a generic edit/readout manifold rather than a named concept module.
