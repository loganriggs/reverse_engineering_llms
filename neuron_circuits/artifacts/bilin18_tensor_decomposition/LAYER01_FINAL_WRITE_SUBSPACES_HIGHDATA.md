# Final-query MLP write subspaces in Layers 0 and 1

Unit-normalized final-position MLP write changes from the original nineteen substitutions and 8 newly selected FineWeb contexts per original lexical pair define frozen family and global SVD bases. Ranks `[4, 8, 16, 24]` are projected into seventeen new substitutions and propagated through the complete suffix. Fixed random bases and wrong-family bases are causal controls.

## Locked hypotheses

- **R0_rank24_global_transfers: not supported.**
- **R1_rank24_family_is_specific: not supported.**
- **R2_rank24_beats_random: not supported.**
- **R3_more_data_rescues_rank6_failure: not supported.**
- **R4_subspace_transfers_ood: not supported.**
- **R5_rank16_is_near_rank24: supported.**

| Layer | Corpus | Rank-24 correct / wrong / advantage (win) | Global | Random | Local correct | Native-256 reference |
|---:|---|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.129 / 0.098 / 0.053 (0.57) | 0.165 | 0.027 | 0.052 | 0.154 |
| 0 | tinystories | 0.031 / 0.051 / 0.013 (0.53) | 0.094 | 0.028 | 0.053 | 0.180 |
| 1 | fineweb | 0.074 / 0.083 / -0.005 (0.49) | 0.098 | 0.015 | 0.051 | 0.065 |
| 1 | tinystories | 0.134 / 0.094 / 0.062 (0.63) | 0.120 | 0.028 | 0.048 | 0.064 |

| Layer | Corpus | Family rank frontier 4 / 8 / 16 / 24 | Global frontier |
|---:|---|---:|---:|
| 0 | fineweb | 0.050 / 0.100 / 0.120 / 0.129 | 0.039 / 0.060 / 0.128 / 0.165 |
| 0 | tinystories | -0.006 / 0.010 / 0.044 / 0.031 | 0.005 / -0.031 / 0.050 / 0.094 |
| 1 | fineweb | 0.010 / 0.023 / 0.071 / 0.074 | 0.044 / 0.047 / 0.092 / 0.098 |
| 1 | tinystories | 0.021 / 0.047 / 0.076 / 0.134 | 0.041 / 0.077 / 0.142 / 0.120 |

A compact global subspace would support a reusable query-position effect interface even when native family banks fail. A family-over-global advantage is the stronger semantic claim; without it, the subspace is a generic edit/readout manifold rather than a named concept module.
