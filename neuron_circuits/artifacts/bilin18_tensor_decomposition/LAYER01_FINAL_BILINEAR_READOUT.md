# Exact conditional bilinear readout at the Layer-0/1 final query

For every prospective lexical edit, the normalized pre-MLP state change is expanded exactly as `ΔL·R(base) + L(base)·ΔR + ΔL·ΔR`. Each term and four channel-selection rules are patched through the complete suffix. Static decoder norm is compared with activation magnitude, state-conditioned write magnitude, and the exact local contribution to the full MLP write.

## Locked hypotheses

- **B0_exact_bilinear_decomposition: supported.**
- **B1_both_cross_terms_are_material: supported.**
- **B2_quadratic_edit_term_is_minor: supported.**
- **B3_state_write_ranking_is_useful: supported.**
- **B4_local_contribution_ranking_recovers_the_readout: not supported.**
- **B5_state_contraction_beats_static_norm: supported.**
- **B6_conditional_program_transfers_ood: supported.**

| Layer | Corpus | Left/right/quadratic projection | Term downstream fidelity | Static / activation / state-write / local-contribution rank-256 |
|---:|---|---:|---:|---:|
| 0 | fineweb | 0.494/0.506/0.000 | 0.576/0.656/0.003 | 0.125/0.640/0.591/0.697 |
| 0 | tinystories | 0.493/0.509/-0.000 | 0.548/0.634/-0.009 | 0.074/0.615/0.576/0.665 |
| 1 | fineweb | 0.507/0.497/-0.001 | 0.587/0.587/-0.006 | 0.023/0.660/0.658/0.691 |
| 1 | tinystories | 0.498/0.501/0.000 | 0.670/0.578/-0.001 | 0.080/0.652/0.644/0.717 |

Maximum product-rule recomposition error: **3.61e-06**.

`local_contribution` is target-aware only at the local MLP-write boundary: it uses the exact full write but no semantic label or downstream logits. `state_write` uses only the realized product change and decoder-column norms, so it is the stronger non-circular test of whether weight contraction identifies active native channels.
