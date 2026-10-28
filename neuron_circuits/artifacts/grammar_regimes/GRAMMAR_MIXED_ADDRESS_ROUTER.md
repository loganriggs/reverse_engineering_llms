# Within-model address router for mixed arbitrary facts

Predictors are fit on disjoint learned facts in template 0 and evaluated on new facts in both template 0 and held-out template 6. The target is the exact final bilinear MLP residual write. Every predicted write is executed through the target model's final RMSNorm and unembedding; effect fidelity is relative to zeroing that natural write. Raw and state predictors never receive the arbitrary answer label. Answer prototypes are explicit leaky diagnostics.

## Locked hypotheses

- **A0_exact_tensor_and_manual_forward_close: supported.**
- **A1_raw_address_does_not_generalize_random_facts: supported.**
- **A2_premlp_linear_state_is_sufficient: not supported.**
- **A3_quadratic_state_repairs_linear_failure: not supported.**
- **A4_memory_is_readable_before_final_attention: not supported.**
- **A5_qk_address_is_sufficient: not supported.**
- **A6_local_state_atlas_is_sufficient: not supported.**
- **A7_selected_facts_are_not_materially_harder_than_rules: supported.**
- **A8_fact_specific_write_residual_is_predictable: not supported.**
- **A9_oracle_answer_prototype_is_a_strong_but_leaky_actuator: not supported.**
- **A10_real_state_beats_shuffled_target_control: supported.**
- **A11_causal_signal_gate_passes: supported.**

| Seed | Status | Template | metadata | raw_tokens | boundary2_full_pca64 | boundary3_final | qk_pattern | attention_write | premlp_linear | premlp_quadratic_16 | premlp_quadratic_32 | premlp_knn_8 | relation_answer_prototype_oracle | exact_bilinear_tensor |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | unselected | 0 | 0.062 | 0.106 | 0.243 | 0.366 | 0.030 | 0.051 | 0.373 | 0.705 | 0.897 | 0.766 | 0.862 | 1.000 |
| 0 | unselected | 6 | 0.061 | 0.025 | 0.222 | 0.331 | 0.004 | 0.035 | 0.347 | 0.673 | 0.880 | 0.734 | 0.830 | 1.000 |
| 0 | selected | 0 | 0.003 | -0.013 | 0.019 | 0.029 | 0.015 | 0.019 | 0.041 | 0.518 | 0.829 | 0.395 | 0.743 | 1.000 |
| 0 | selected | 6 | 0.004 | -0.009 | 0.008 | 0.030 | -0.001 | 0.021 | 0.032 | 0.500 | 0.797 | 0.400 | 0.663 | 1.000 |
| 101 | unselected | 0 | 0.089 | 0.141 | 0.402 | 0.477 | 0.101 | 0.163 | 0.456 | 0.746 | 0.886 | 0.810 | 0.868 | 1.000 |
| 101 | unselected | 6 | 0.095 | 0.053 | 0.290 | 0.435 | -0.079 | 0.141 | 0.420 | 0.724 | 0.862 | 0.775 | 0.831 | 1.000 |
| 101 | selected | 0 | 0.006 | -0.020 | 0.009 | 0.007 | 0.006 | 0.002 | 0.017 | 0.529 | 0.789 | 0.451 | 0.726 | 1.000 |
| 101 | selected | 6 | 0.001 | -0.031 | 0.012 | -0.010 | 0.003 | -0.008 | -0.005 | 0.510 | 0.783 | 0.433 | 0.662 | 1.000 |
| 104 | unselected | 0 | 0.045 | 0.076 | 0.347 | 0.472 | 0.114 | 0.112 | 0.473 | 0.758 | 0.907 | 0.783 | 0.842 | 1.000 |
| 104 | unselected | 6 | 0.046 | -0.000 | 0.213 | 0.445 | 0.012 | 0.107 | 0.462 | 0.703 | 0.902 | 0.767 | 0.805 | 1.000 |
| 104 | selected | 0 | -0.014 | -0.043 | -0.001 | 0.052 | -0.019 | -0.016 | 0.065 | 0.528 | 0.849 | 0.461 | 0.743 | 1.000 |
| 104 | selected | 6 | -0.021 | -0.052 | -0.018 | 0.051 | -0.041 | -0.050 | 0.069 | 0.506 | 0.848 | 0.454 | 0.686 | 1.000 |

Write R2 asks for coordinate reconstruction; oracle-residual R2 subtracts the relation+answer prototype and asks whether fact-specific variation remains predictable. Finite effect fidelity asks whether the predicted write causes the same answer-distribution change. Accuracy asks whether that change preserves the learned answer. These are deliberately reported separately.
