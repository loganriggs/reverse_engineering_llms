# Coordinated-greedy red team of native Layer-16 loss breadth

The complete discovery contribution Gram matrix is used to add the channel with the largest exact residual-error reduction at each step. This accounts for cancellations among already selected channels, unlike marginal ranking. The ordering is frozen before disjoint corpus evaluation; the discovery curve is shown alongside held-out performance to expose selection overfit.

## Locked hypotheses

- **G0_greedy_discovery_error_decreases_monotonically: not supported.**
- **G1_greedy_512_channels_are_loss_sufficient: not supported.**
- **G2_greedy_materially_beats_marginal: not supported.**
- **G3_greedy_2048_channels_are_high_fidelity: not supported.**
- **G4_greedy_bank_transfers_ood: supported.**
- **G5_greedy_native_k90_is_at_most_1024: not supported.**
- **G6_coordination_reorders_channels: not supported.**

Top-512 overlap with marginal ordering: `325/512`. Discovery monotonic: `False`.

| Corpus | K | Discovery greedy | Held-out greedy | Held-out marginal |
|---|---:|---:|---:|---:|
| fineweb | 32 | 0.536 | 0.604 | 0.576 |
| fineweb | 128 | 0.698 | 0.719 | 0.717 |
| fineweb | 512 | 0.855 | 0.768 | 0.763 |
| fineweb | 1024 | 0.931 | 0.804 | 0.797 |
| fineweb | 2048 | 0.980 | 0.881 | 0.870 |
| fineweb | 4096 | 0.993 | 0.974 | 0.970 |
| fineweb | 4608 | 1.000 | 1.000 | 1.000 |
| fineweb | K90 | — | 4096 | 4096 |
| wikitext | 32 | 0.536 | 0.572 | 0.541 |
| wikitext | 128 | 0.698 | 0.692 | 0.698 |
| wikitext | 512 | 0.855 | 0.756 | 0.753 |
| wikitext | 1024 | 0.931 | 0.797 | 0.788 |
| wikitext | 2048 | 0.980 | 0.867 | 0.859 |
| wikitext | 4096 | 0.993 | 0.969 | 0.966 |
| wikitext | 4608 | 1.000 | 1.000 | 1.000 |
| wikitext | K90 | — | 4096 | 4096 |
| tinystories | 32 | 0.536 | 0.514 | 0.465 |
| tinystories | 128 | 0.698 | 0.702 | 0.712 |
| tinystories | 512 | 0.855 | 0.772 | 0.778 |
| tinystories | 1024 | 0.931 | 0.814 | 0.815 |
| tinystories | 2048 | 0.980 | 0.878 | 0.879 |
| tinystories | 4096 | 0.993 | 0.965 | 0.974 |
| tinystories | 4608 | 1.000 | 1.000 | 1.000 |
| tinystories | K90 | — | 4096 | 4096 |

Coordinated greedy is exact for its one-step criterion but is not a global subset optimizer. A large discovery/held-out gap would weaken any sparsity conclusion from this selector; similar held-out breadth under both rankings would strengthen the conclusion that no small stable native bank captures this loss family.
