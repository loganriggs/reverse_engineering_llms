# Origin of the generic L17H2 edit mode

Post-boundary normalized residual deltas and native H2 current-value deltas are compared for spelling and sixteen unrelated token swaps. Split-0 axes are frozen and evaluated on split 1; the native H2 c_v slice then maps the residual axes into value space without fitting.

## Locked results

- **R0_edit_mode_preexists_value_projection: not supported.**
- **R1_prevalue_mode_is_generic: not supported.**
- **R2_weights_map_residual_axis_to_value_axis: supported.**
- **R3_value_projection_canonicalizes_the_mode: supported.**
- **R4_global_weight_outlier_is_the_data_mode: supported.**
- **R5_weight_predictable_generic_edit_bus: not supported.**

| Metric | Value |
|---|---:|
| spelling_residual_r1 | 0.2073 |
| spelling_value_r1 | 0.7787 |
| control_residual_r1 | 0.0679 |
| control_value_r1 | 0.3899 |
| residual_axis_cosine | 0.6060 |
| value_axis_cosine | 0.8952 |
| spelling_mapped_residual_to_value_cosine | 0.9995 |
| control_mapped_residual_to_value_cosine | 0.9907 |
| weight_top_output_spelling_value_cosine | 0.9554 |
| weight_top_output_control_value_cosine | 0.8058 |
| weight_top_input_spelling_residual_cosine | 0.8206 |

## Interpretation boundary

A shared residual axis would locate the edit bus before H2. Strong native weight mapping but weak global singular-vector alignment would mean c_v faithfully transforms a data-conditioned mode rather than selecting its largest isotropic weight mode. Canonicalization requires a genuine increase in held-out rank-1 energy, not merely high axis cosine.
