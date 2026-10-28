# Native-channel sparsity inside the compact 546M late-MLP operator

At each of Layers 13–16, the exact local bilinear derivative on a frozen natural rank-32 input chart is expanded into all 4,608 native CP channels. A static bank is ranked only on FineWeb by its exact individual reduction in derivative error, then tested on disjoint FineWeb, WikiText, and TinyStories. The per-example bank is a hindsight ceiling. This is a local operator test, not yet an end-to-end suffix intervention.

## Locked hypotheses

- **N0_native_channel_sum_is_exact: supported.**
- **N1_static_512_channels_are_sufficient: not supported.**
- **N2_static_1024_channels_are_high_fidelity: not supported.**
- **N3_ranked_channels_beat_random: not supported.**
- **N4_context_specific_channel_selection_is_material: supported.**
- **N5_static_bank_transfers_ood: supported.**
- **N6_native_neuron_sparsity_is_sufficient: not supported.**
- **N7_some_local_operators_are_broad: supported.**
- **N8_factor_swap_is_invariant: supported.**

Left/right factor-swap relative error: `0.00e+00`.

| Corpus | Layer | Static 128 / 512 / 1024 / 2048 | Oracle512 | Random512 p95 | K90 |
|---|---:|---:|---:|---:|---:|
| fineweb | 13 | 0.131 / 0.249 / 0.382 / 0.602 | 0.478 | 0.153 | 4096 |
| fineweb | 14 | 0.100 / 0.220 / 0.355 / 0.581 | 0.456 | 0.138 | 4096 |
| fineweb | 15 | 0.294 / 0.396 / 0.502 / 0.679 | 0.572 | 0.180 | 4096 |
| fineweb | 16 | 0.802 / 0.840 / 0.870 / 0.917 | 0.870 | 0.386 | 2048 |
| wikitext | 13 | 0.138 / 0.260 / 0.391 / 0.611 | 0.484 | 0.146 | 4096 |
| wikitext | 14 | 0.102 / 0.224 / 0.357 / 0.584 | 0.460 | 0.133 | 4096 |
| wikitext | 15 | 0.300 / 0.404 / 0.510 / 0.685 | 0.576 | 0.235 | 4096 |
| wikitext | 16 | 0.821 / 0.856 / 0.883 / 0.925 | 0.882 | 0.250 | 2048 |
| tinystories | 13 | 0.142 / 0.265 / 0.396 / 0.612 | 0.490 | 0.144 | 4096 |
| tinystories | 14 | 0.108 / 0.233 / 0.367 / 0.589 | 0.463 | 0.140 | 4096 |
| tinystories | 15 | 0.313 / 0.421 / 0.525 / 0.696 | 0.587 | 0.168 | 4096 |
| tinystories | 16 | 0.817 / 0.857 / 0.885 / 0.927 | 0.884 | 0.282 | 2048 |

A short `I/B` path description can still execute a dense native operator. Path-type sparsity and neuron sparsity answer different questions: the former compresses algebraic interaction order; the latter asks whether the operator itself is localized in the architecture's CP basis.
