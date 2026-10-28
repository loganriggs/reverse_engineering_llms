# Downstream relevance of the Layer-16 common operator and contextual correction

The exact Layer-16 local MLP derivative is composed site-by-site with the Layer-17 residual scale, RMS-normalization Jacobian, and a frozen natural rank-32 output chart. Attention is fixed at its native response, matching the established MLP path boundary. This distinguishes locally large variation from variation that reaches the output-facing state.

## Locked hypotheses

- **D0_composed_endpoint_derivative_is_exact: supported.**
- **D1_downstream_readout_compresses_contextual_variation: supported.**
- **D2_endpoint_rank16_atlas_is_sufficient: supported.**
- **D3_endpoint_rank64_atlas_is_high_fidelity: supported.**
- **D4_common_local_operator_is_behaviorally_sufficient: supported.**
- **D5_factorized_local_program_is_behaviorally_sufficient: supported.**
- **D6_inner_rank8_preserves_the_causal_atlas: supported.**
- **D7_downstream_program_transfers_ood: supported.**
- **D8_contextual_correction_is_behaviorally_material: supported.**

Independent autograd relative error: `4.69e-07`.
Discovery endpoint spectrum: uncentered R90 `2`, centered R90 `26`, centered top share `0.338`.

| Corpus | Endpoint spectral R8 / R16 / R64 | Mean local | Full local64 | Local64 inner R1 / R2 / R4 / R8 |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.970 / 0.977 / 0.985 | 0.868 | 0.986 | 0.590 / 0.878 / 0.962 / 0.978 |
| wikitext | 0.974 / 0.979 / 0.988 | 0.908 | 0.988 | 0.599 / 0.879 / 0.965 / 0.981 |
| tinystories | 0.973 / 0.979 / 0.988 | 0.905 | 0.989 | 0.611 / 0.871 / 0.964 / 0.981 |

The endpoint spectral atlas is a hindsight coordinate bound. The local programs are executable: their coefficients come from exact weight-derived state reads, and the site-specific downstream RMS map is computed from the native state. Neither result yet assigns natural-language semantics to the reads.
