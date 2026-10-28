# Exact native-channel frontier in the final bilinear block

Each final-block bilinear channel contributes one exact rank-one residual operator, including both product-rule arms. Channels are ranked once by mean squared complete Layer-0 answer effect on discovery prompts, then retained or removed on held-out prompts. The identity/QK/value path remains intact.

## Locked hypotheses

- **G0_all_channels_recompose_exactly: supported.**
- **G1_pure_rule_has_a_compact_native_bank: supported.**
- **G2_mixed_top32_is_insufficient: supported.**
- **G3_mixed_top128_is_sufficient: supported.**
- **G4_native_ranking_beats_random_channels: not supported.**
- **G5_mixed_bank_is_globally_reusable: not supported.**
- **G6_top_channels_are_necessary: supported.**

| Seed | rho | retain 0/16/32/64/128/192 | random32 p95 | site top32 | remove top32 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.644/0.973/0.996/1.000/1.000/1.000 | 0.825 | 0.999 | 0.700 |
| 0 | 0.5 | 0.180/0.367/0.491/0.694/0.928/1.000 | 0.408 | 0.733 | 0.874 |
| 1 | 0.0 | 0.673/0.871/0.941/0.981/0.999/1.000 | 0.787 | 0.976 | 0.839 |
| 1 | 0.5 | 0.201/0.392/0.515/0.712/0.938/1.000 | 0.429 | 0.752 | 0.870 |

Retain-only fidelity measures sufficiency of a fixed native bank inside the exact tangent program. Removal measures necessity in that same linearized chain. Per-site top-32 is an importance oracle, not a deployable explanation, and can still be nonoptimal under cancellation.
