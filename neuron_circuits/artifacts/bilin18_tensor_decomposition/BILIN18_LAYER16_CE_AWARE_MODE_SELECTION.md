# CE-aware selection from the frozen Layer-16 operator atlas

The local operator modes and exact state read vectors are unchanged. A fixed bank is selected only by its contribution to true-next-token CE derivatives on discovery FineWeb, using exact reverse-mode gradients from the Layer-16 write through the remaining model. This is metric-aware mode selection, not per-example fitting.

## Locked hypotheses

- **B0_reverse_ce_derivative_matches_forward_jvp: supported.**
- **B1_ce_aware_64_mode_program_is_sufficient: not supported.**
- **B2_ce_aware_selection_beats_energy_prefix: not supported.**
- **B3_ce_aware_128_mode_program_is_high_fidelity: not supported.**
- **B4_ce_aware_selection_transfers_ood: supported.**
- **B5_ce_aware_selection_beats_random: supported.**
- **B6_top256_local_modes_cover_loss: not supported.**
- **B7_loss_relevance_reorders_local_energy_modes: not supported.**

Forward/reverse CE derivative relative error: `8.09e-07`. Top-64 overlap with local-energy prefix: `46/64`.
On selection discovery, K64 CE-aware/energy-prefix fidelity is `0.782` / `0.758`.

| Corpus | K | CE-aware | Energy prefix | Random p95 (K64) |
|---|---:|---:|---:|---:|
| fineweb | 8 | 0.705 | 0.723 | — |
| fineweb | 16 | 0.751 | 0.751 | — |
| fineweb | 32 | 0.773 | 0.777 | — |
| fineweb | 64 | 0.801 | 0.803 | 0.631 |
| fineweb | 128 | 0.829 | 0.833 | — |
| fineweb | 256 | 0.870 | 0.870 | — |
| wikitext | 8 | 0.696 | 0.701 | — |
| wikitext | 16 | 0.721 | 0.733 | — |
| wikitext | 32 | 0.739 | 0.757 | — |
| wikitext | 64 | 0.768 | 0.779 | 0.593 |
| wikitext | 128 | 0.802 | 0.813 | — |
| wikitext | 256 | 0.845 | 0.845 | — |
| tinystories | 8 | 0.714 | 0.723 | — |
| tinystories | 16 | 0.764 | 0.768 | — |
| tinystories | 32 | 0.783 | 0.804 | — |
| tinystories | 64 | 0.820 | 0.838 | 0.444 |
| tinystories | 128 | 0.861 | 0.867 | — |
| tinystories | 256 | 0.904 | 0.904 | — |

Selection can improve one quotient while harming another. These banks are justified for local true-token loss derivatives only; they are not automatically the best state, full-logit, Fisher, or semantic basis. The 256-mode ceiling also tests whether the discarded local spectral tail carries disproportionate loss relevance.
