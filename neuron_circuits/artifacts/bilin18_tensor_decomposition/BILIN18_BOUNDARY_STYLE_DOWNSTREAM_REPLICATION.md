# Held-out replication of downstream spelling consumers

L17H2, complete Layer-17 destination attention, and L11H6 were selected from the fold-0/split-0 screen. The complete 81-edge screen is rerun on the other five fold/split cells, but only these locked claims are evaluated.

## Locked results

- **R0_native_endpoints_are_consistent: supported.**
- **R1_L17H2_boundary_replicates: supported.**
- **R2_L17_full_destination_replicates: supported.**
- **R3_L11H6_is_weak_not_major: supported.**
- **R4_boundary_only_is_incomplete_at_L17: supported.**
- **R5_residual_consumer_is_distributed: supported.**

| Fold/split | L9H8 | L17H2 | L17 all-boundary | L17 full-destination | L11H6 |
|---|---:|---:|---:|---:|---:|
| 0/1 | 0.541 | 0.080 | 0.080 | 0.293 | 0.020 |
| 1/0 | 0.522 | 0.083 | 0.083 | 0.305 | 0.019 |
| 1/1 | 0.546 | 0.090 | 0.090 | 0.272 | 0.021 |
| 2/0 | 0.515 | 0.096 | 0.096 | 0.314 | 0.022 |
| 2/1 | 0.536 | 0.089 | 0.089 | 0.292 | 0.025 |

## Interpretation boundary

Replication of complete destination attention without comparable boundary-edge mediation means the suffix consumes the spelling edit through a broader contextual attention update. It does not by itself distinguish many individually tiny causal edges from an interaction or a changed destination query. The final-layer tensor factors and additive residual branches are the next localization boundary.
