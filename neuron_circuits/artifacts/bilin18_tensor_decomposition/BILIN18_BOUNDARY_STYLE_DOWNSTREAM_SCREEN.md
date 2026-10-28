# Screen of downstream consumers of the L8H2 spelling edit

After the finite L8H2 cue-edge replacement, every boundary-to-destination edge in Layers 9--17 is individually frozen to its pre-edit write and propagated through the full suffix. All boundary edges and the complete destination-attention change are additional layer controls. This discovery screen uses one locked spelling fold/split; selected routes require disjoint replication.

## Locked results

- **S0_native_endpoints_are_consistent: supported.**
- **S1_known_L9H8_is_rediscovered: supported.**
- **S2_additional_later_consumer_exists: not supported.**
- **S3_boundary_edges_explain_each_attention_stage: not supported.**
- **S4_destination_attention_has_nonboundary_residual: not supported.**

| Rank | Edge | mediated fraction | mean | win | random percentile |
|---:|---|---:|---:|---:|---:|
| 1 | L9H8 | 0.543 | +0.04404 | 0.914 | 1.000 |
| 2 | L17H2 | 0.085 | +0.006859 | 0.898 | 1.000 |
| 3 | L11H6 | 0.018 | +0.00148 | 0.859 | 1.000 |
| 4 | L15H1 | 0.007 | +0.0005527 | 0.828 | 1.000 |
| 5 | L11H1 | 0.005 | +0.0004295 | 0.641 | 1.000 |
| 6 | L12H4 | 0.005 | +0.0003749 | 0.711 | 1.000 |
| 7 | L12H3 | 0.004 | +0.0003171 | 0.594 | 1.000 |
| 8 | L10H2 | 0.003 | +0.0002805 | 0.828 | 1.000 |
| 9 | L11H5 | 0.003 | +0.0002607 | 0.789 | 1.000 |
| 10 | L10H5 | 0.002 | +0.0001397 | 0.719 | 1.000 |
| 11 | L10H6 | 0.001 | +6.361e-05 | 0.617 | 0.871 |
| 12 | L14H2 | 0.001 | +5.852e-05 | 0.688 | 0.968 |
| 13 | L11H0 | 0.001 | +5.637e-05 | 0.680 | 0.774 |
| 14 | L12H2 | 0.001 | +5.201e-05 | 0.672 | 1.000 |
| 15 | L11H4 | 0.001 | +4.494e-05 | 0.570 | 1.000 |
| 16 | L15H2 | 0.000 | +3.998e-05 | 0.578 | 1.000 |

| Layer | best head | all boundary | full destination attention |
|---:|---:|---:|---:|
| 9 | 0.543 | 0.544 | 0.544 |
| 10 | 0.003 | 0.007 | 0.028 |
| 11 | 0.018 | 0.028 | 0.058 |
| 12 | 0.005 | 0.010 | 0.027 |
| 13 | 0.000 | -0.000 | 0.002 |
| 14 | 0.001 | 0.001 | 0.004 |
| 15 | 0.007 | 0.007 | 0.045 |
| 16 | 0.000 | 0.000 | -0.000 |
| 17 | 0.085 | 0.084 | 0.284 |

## Interpretation boundary

Individual freeze effects are non-additive and can be negative. This screen ranks candidate consumers; it does not assign percentages that sum to one. A later edge can redundantly read a state already consumed earlier, and freezing an earlier edge changes the states presented to later routes. The next gate is a prospectively frozen multi-edge program on disjoint folds and splits.
