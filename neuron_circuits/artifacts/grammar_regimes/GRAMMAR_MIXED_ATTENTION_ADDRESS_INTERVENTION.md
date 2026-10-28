# Finite final-attention interventions on the mixed address code

Each candidate replaces the final attention residual at held-out template-6 facts. The exact bilinear MLP is recomputed. Decoder fidelity holds the native post-attention residual fixed and compares only the induced MLP write; complete-route fidelity includes the replaced attention residual itself. Answer prototypes are explicitly leaky diagnostics.

## Locked hypotheses

- **C0_native_recomposition_is_exact: supported.**
- **C1_upstream_address_survives_attention_deletion: supported.**
- **C2_attention_is_needed_for_complete_route: not supported.**
- **C3_arbitrary_facts_need_attention_more_than_rules: not supported.**
- **C4_relation_mean_attention_is_sufficient: supported.**
- **C5_answer_oracle_attention_is_sufficient: supported.**
- **C6_site_specific_attention_matters: not supported.**
- **C7_attention_is_a_small_norm_large_effect_correction: not supported.**
- **C8_causal_signal_gate_passes: supported.**

| Seed | Status | native | delete_attention | global_prototype | relation_prototype | answer_prototype | relation_answer_prototype | shuffled_attention |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | unselected | 1.000 | 0.955 | 0.956 | 0.957 | 0.959 | 0.958 | 0.901 |
| 0 | selected | 1.000 | 0.914 | 0.915 | 0.916 | 0.922 | 0.921 | 0.842 |
| 101 | unselected | 1.000 | 0.898 | 0.904 | 0.900 | 0.907 | 0.907 | 0.790 |
| 101 | selected | 1.000 | 0.778 | 0.783 | 0.778 | 0.813 | 0.811 | 0.600 |
| 104 | unselected | 1.000 | 0.924 | 0.927 | 0.928 | 0.933 | 0.941 | 0.861 |
| 104 | selected | 1.000 | 0.907 | 0.914 | 0.912 | 0.915 | 0.912 | 0.795 |

Cells report complete-route Fisher fidelity. Decoder-only fidelity is retained in JSON to distinguish whether attention changes the bilinear address versus contributing a direct residual write.
