# Native Layer-16 channel support of true-token loss derivatives

Each exact CE directional derivative is decomposed into the 4,608 native bilinear-channel contributions: the channel's local gate derivative times the exact reverse-mode sensitivity of its Down write. A static marginal bank is ranked on discovery FineWeb and frozen for disjoint evaluation. The per-example bank is a marginal heuristic, not a ceiling or combinatorial optimum.

## Locked hypotheses

- **Q0_native_channel_ce_sum_is_exact: supported.**
- **Q1_static_512_channels_are_loss_sufficient: not supported.**
- **Q2_static_1024_channels_are_high_fidelity: not supported.**
- **Q3_ranked_channels_beat_random: supported.**
- **Q4_native_loss_neuron_sparsity_is_sufficient: not supported.**
- **Q5_static_loss_bank_transfers_ood: supported.**
- **Q6_context_specific_channel_selection_is_material: not supported.**
- **Q7_loss_is_sparser_than_local_operator_energy: not supported.**
- **Q8_full_native_bank_has_unit_fidelity: supported.**

Maximum all-channel reconstruction error: `2.73e-07`.

| Corpus | Static 128 / 512 / 1024 / 2048 | Oracle512 | Random512 p95 | K90 |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.717 / 0.763 / 0.797 / 0.870 | 0.402 | 0.285 | 4096 |
| wikitext | 0.698 / 0.753 / 0.788 / 0.859 | 0.334 | 0.269 | 4096 |
| tinystories | 0.712 / 0.778 / 0.815 / 0.879 | 0.416 | 0.272 | 4096 |

This is native-basis behavioral sparsity, not semantic monosemanticity. A channel can contribute coherently to many unrelated next-token losses, and marginal ranking can miss coordinated cancelling subsets. It directly answers whether a small fixed neuron bank preserves this loss derivative family.
