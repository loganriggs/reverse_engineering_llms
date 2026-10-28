# CE relevance in the near-full Layer-16 spectral tail

The local atlas is enlarged from 256 to 1,024 uncentered modes using 1,024 FineWeb position-sites. Every mode keeps its exact weight-derived state coefficient. Fixed CE-aware selection and the ordinary local-energy prefix are evaluated on disjoint 128-site panels.

## Locked hypotheses

- **T0_reverse_ce_derivative_matches_forward_jvp: supported.**
- **T1_energy_prefix512_is_sufficient: not supported.**
- **T2_near_full_1024_atlas_is_high_fidelity: not supported.**
- **T3_ce_aware_256_is_sufficient: not supported.**
- **T4_ce_aware_512_is_high_fidelity: not supported.**
- **T5_loss_k90_is_at_most_256: not supported.**
- **T6_tail_program_transfers_ood: supported.**
- **T7_loss_relevance_strongly_reorders_top256: not supported.**
- **T8_spectral_tail_is_behaviorally_material: supported.**

Forward/reverse error: `8.09e-07`. CE-aware/energy top-256 overlap: `140/256`. Atlas discovery energy R90: `69`.
Discovery K256 CE-aware/energy: `0.911` / `0.850`.

| Corpus | K | CE-aware | Energy prefix |
|---|---:|---:|---:|
| fineweb | 64 | 0.799 | 0.819 |
| fineweb | 128 | 0.825 | 0.847 |
| fineweb | 256 | 0.861 | 0.884 |
| fineweb | 512 | 0.907 | 0.926 |
| fineweb | 768 | 0.951 | 0.958 |
| fineweb | 1024 | 0.988 | 0.988 |
| fineweb | K90 | 512 | — |
| wikitext | 64 | 0.778 | 0.787 |
| wikitext | 128 | 0.803 | 0.818 |
| wikitext | 256 | 0.837 | 0.856 |
| wikitext | 512 | 0.892 | 0.918 |
| wikitext | 768 | 0.940 | 0.955 |
| wikitext | 1024 | 0.985 | 0.985 |
| wikitext | K90 | 768 | — |
| tinystories | 64 | 0.836 | 0.852 |
| tinystories | 128 | 0.858 | 0.886 |
| tinystories | 256 | 0.893 | 0.916 |
| tinystories | 512 | 0.930 | 0.953 |
| tinystories | 768 | 0.961 | 0.975 |
| tinystories | 1024 | 0.991 | 0.991 |
| tinystories | K90 | 512 | — |

If fidelity rises mainly by including the tail rather than by reordering a small bank, the loss computation is genuinely broad in this operator basis. That still does not imply 1,024 semantic mechanisms: modes are dense coordinates, and the endpoint is a scalar training-loss derivative rather than all behaviors.
