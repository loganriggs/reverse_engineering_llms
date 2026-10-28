# Compact boundary-write module and state router

A fixed FineWeb SVD basis of exact Layer-11 newline write changes is evaluated with oracle coefficients and with coefficients predicted from upstream state changes. Equal-rank random bases and shuffled routers are controls.

## Locked hypotheses

- **S0_rank16_oracle_subspace_is_sufficient: not supported.**
- **S1_compact_frontier_reaches_ood: supported.**
- **S2_learned_subspace_beats_random: supported.**
- **S3_state_router_recovers_compact_module: not supported.**
- **S4_router_beats_shuffled_coefficients: not supported.**
- **S5_subspace_repairs_wikitext_direction_rotation: supported.**

Primary rank `16`; selected ridge penalty `1`; calibration coefficient MSE `1.552e+05`.

| Corpus | Rank-16 oracle/random fidelity | Routed/shuffled fidelity | Rank-16 energy | Smallest rank with fidelity >=0.80 |
|---|---:|---:|---:|---:|
| fineweb | 0.863/-0.071 | 0.830/0.722 | 0.969 | 4 |
| wikitext | 0.786/-0.105 | 0.587/0.361 | 0.949 | 32 |
| tinystories | 0.853/-0.099 | 0.842/0.680 | 0.970 | 8 |

Oracle projection asks whether a fixed compact dictionary contains the needed directions. The router asks the separate question of whether a simple upstream-state map selects the right combination.
