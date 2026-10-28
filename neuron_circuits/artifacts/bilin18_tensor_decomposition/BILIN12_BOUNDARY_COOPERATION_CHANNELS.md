# Cooperative bilinear terms and native-channel frontier

The shuffled-base correction evaluates shuffled cross terms together with the intact edit-squared term. Native channel sets are ranked on disjoint FineWeb discovery examples and compared with random and per-example oracle sets.

## Locked hypotheses

- **C0_correct_base_pairing_is_necessary_in_full_sum: not supported.**
- **C1_cross_and_quadratic_writes_cancel: not supported.**
- **C2_cancellation_is_large: not supported.**
- **P0_fixed_128_channels_nearly_recover_module: supported.**
- **P1_fixed_32_channels_are_useful: supported.**
- **P2_weighted_native_selection_beats_random: supported.**
- **P3_contextual_channel_selection_matters: not supported.**

| Corpus | Full/shuffled-full fidelity | Cross–quadratic cosine | Cancellation ratio | Fixed/random/oracle 32 fidelity | Fixed 128 fidelity |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.883/0.700 | -0.455 | 0.533 | 0.736/0.003/0.753 | 0.836 |
| wikitext | 0.812/0.543 | -0.363 | 0.627 | 0.693/0.003/0.708 | 0.768 |
| tinystories | 0.870/0.690 | -0.480 | 0.523 | 0.745/0.004/0.771 | 0.819 |

The fixed sets test a reusable native module. The oracle sets test context-dependent multiplexing at the same channel budget. Neither metric by itself assigns a human meaning to individual channels.
