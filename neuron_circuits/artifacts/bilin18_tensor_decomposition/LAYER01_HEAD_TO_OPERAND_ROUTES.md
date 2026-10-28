# Attention-head and carried-state routes into bilinear operands

Each native head's exact final-position attention delta, plus Layer-1 carried-state change, is propagated through the analytic RMSNorm Jacobian and integrated along the straight finite edit path. The resulting source effects recompose every left/right factor endpoint. For each causally selected neuron, attribution follows whichever factor supplies its dominant cross term.

## Locked hypotheses

- **M0_integrated_sources_recompose_factors: not supported.**
- **M1_three_sources_explain_each_operand: supported.**
- **M2_frozen_transport_heads_cover_operands: not supported.**
- **M3_layer0_source_roles_are_more_recurrent: supported.**
- **M4_layer0_source_roles_transfer_ood: supported.**
- **M5_layer1_carried_state_is_material: supported.**
- **M6_causal_selection_enriches_source_concentration: not supported.**

| Layer | Corpus | Top-3 source share | Frozen-head share | Carried share | Cancellation ratio | Random top-3 | Recurring source agreement |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.867 | 0.712 | 0.000 | 1.222 | 0.832 | 0.714 (816) |
| 0 | tinystories | 0.856 | 0.626 | 0.000 | 1.243 | 0.805 | 0.667 (834) |
| 1 | fineweb | 0.764 | 0.446 | 0.170 | 1.336 | 0.715 | 0.500 (783) |
| 1 | tinystories | 0.746 | 0.381 | 0.191 | 1.372 | 0.702 | 0.500 (808) |

| Layer | Cross-corpus dominant-source agreement | Frozen heads | Top weighted source→channel edges |
|---:|---:|---:|---:|
| 0 | 0.658 | [7, 4, 2] | head1→3844, head7→1193, head7→3790, head7→2264, head7→292, head7→3806 |
| 1 | 0.400 | [8, 0, 2] | head8→2501, head4→3070, head4→930, head4→2549, head0→3696, head2→261 |

Maximum integrated factor recomposition error: **7.14e-05**. Maximum discrepancy from native float32 attention accumulation/end state: **5.66e-05**.

## Concrete routes

### Layer 0
- ** great-> terrible**: N594 right←head7 (0.81); N1193 right←head7 (0.80); N2312 right←head7 (0.73). Context: ` awaits the redeeming word.
Thus the first great novel of world literature stands at the beginning of the time when the Christian God began`
- ** king-> queen**: N1737 right←head7 (0.60); N1145 right←head7 (0.67); N2264 right←head7 (0.42). Context: ` Because he can. Because he’s the king and has the power to do it.
This is what the prophets warned about`
- ** do-> did**: N1193 right←head7 (0.84); N145 left←head7 (0.80); N1997 right←head7 (0.78). Context: ` diplomats than men."In a lot of ways we do have advantages," she said. "Diplomacy is about being able to put`
- ** cold-> hot**: N1492 right←head7 (0.69); N1737 right←head7 (0.57); N1257 right←head7 (0.67). Context: ` gained a lot of weight, the winter and the cold was extremely difficult for him.
He lost a lot weight again even though his`
### Layer 1
- ** great-> terrible**: N4190 right←head8 (0.58); N2695 left←head8 (0.58); N959 right←head8 (0.44). Context: ` awaits the redeeming word.
Thus the first great novel of world literature stands at the beginning of the time when the Christian God began`
- ** king-> queen**: N1474 right←carried (0.29); N2198 right←head5 (0.34); N3851 left←head5 (0.22). Context: ` Because he can. Because he’s the king and has the power to do it.
This is what the prophets warned about`
- ** do-> did**: N419 right←carried (0.76); N1300 left←carried (0.80); N4002 right←carried (0.34). Context: ` diplomats than men."In a lot of ways we do have advantages," she said. "Diplomacy is about being able to put`
- ** cold-> hot**: N2388 left←head4 (0.26); N2501 left←head8 (0.30); N3001 right←head3 (0.22). Context: ` gained a lot of weight, the winter and the cold was extremely difficult for him.
He lost a lot weight again even though his`

Absolute source shares measure concentration but can hide cancellation; the cancellation ratio is reported alongside them. Head identities are native architectural coordinates, while the straight-path allocation is one principled finite attribution rather than a claim of path uniqueness.
