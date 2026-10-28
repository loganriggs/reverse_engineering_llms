# QK-versus-value factor program for 546M late attention

Every single-layer attention response is split exactly into the symmetry-invariant sum of both normalized two-QK routing branches (`Q`) and current-value/OV response (`V`). The persistent Layer-0 value bus is fixed because the intervention begins after Layer 12. Identity plus ten layer-factor paths is compared with the complete 32-path analytic target, so missing multi-attention-layer interactions remain visible.

## Locked hypotheses

- **F0_qk_value_singletons_are_exact: supported.**
- **F1_singletons_explain_attention_program: supported.**
- **F2_four_factor_paths_are_high_fidelity: supported.**
- **F3_qk_routing_dominates_value_response: supported.**
- **F4_factor_paths_beat_random: not supported.**
- **F5_factor_program_transfers_ood: supported.**
- **F6_compact_factor_and_mlp_program_predicts_finite: supported.**
- **F7_context_specific_factor_selection_is_small: supported.**
- **F8_qk_branch_asymmetry_is_small: supported.**

Maximum singleton factorization error: `1.22e-06`.

Frozen factor order: `I`, `L17Q`, `L13Q`, `L14Q`, `L15Q`, `L15V`, `L16Q`, `L17V`, `L13V`, `L14V`, `L16V`.

| Corpus | All singletons | Q-only / V-only | K4 frozen / oracle / random-p95 | MLP4 + factor K4→finite | Q1/Q2 norm |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.999 | 0.999 / 0.976 | 0.998 / 0.998 / 0.996 | 0.953 | 1.22 |
| wikitext | 0.998 | 0.998 / 0.972 | 0.996 / 0.997 / 0.994 | 0.947 | 1.01 |
| tinystories | 0.999 | 0.999 / 0.971 | 0.997 / 0.998 / 0.996 | 0.961 | 1.25 |

`Q` is canonical under exchanging the two QK factor branches; Q1 and Q2 are reported only as a symmetry/red-team diagnostic. A compact Q/V program identifies an operator role, not a human semantic concept or a fixed head circuit.
