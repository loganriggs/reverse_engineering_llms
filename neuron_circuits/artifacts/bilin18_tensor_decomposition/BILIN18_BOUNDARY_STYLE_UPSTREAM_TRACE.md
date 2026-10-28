# Upstream construction of the L9H8 boundary style value

The cue swap leaves the boundary token unchanged. The Layer-9 mixed-state difference at that boundary is telescoped exactly into initial/repeated embeddings and every earlier attention/MLP write. A 32-point integrated RMSNorm/value-map decomposition then assigns the finite L9H8 current-value change to those additive sources. Each assembled value is inserted into the single boundary edge and propagated through the native suffix.

## Locked results

- **U0_telescope_and_value_decomposition_recompose: supported.**
- **U1_repeated_embedding_is_negative_control: supported.**
- **U2_no_single_layer_creates_the_value: not supported.**
- **U3_attention_and_mlp_are_both_material: not supported.**
- **U4_style_value_is_compiled_by_layer2: not supported.**
- **U5_style_value_is_compiled_by_layer6: not supported.**
- **U6_full_trace_preserves_direct_transport: supported.**

| Fold/split | Full win/mean | Attention mean | MLP mean | Prefix L2/L6 mean | strongest layer/share |
|---|---:|---:|---:|---:|---:|
| 0/0 | 0.875/+0.05628 | +0.05903 | -0.00257 | +9.454e-05/+0.01653 | L8/0.733 |
| 0/1 | 0.859/+0.06444 | +0.05588 | +0.008521 | +7.389e-05/+0.01373 | L8/0.793 |
| 1/0 | 0.930/+0.06525 | +0.06043 | +0.004779 | +6.909e-05/+0.01405 | L8/0.785 |
| 1/1 | 0.844/+0.05084 | +0.05355 | -0.002684 | +6.807e-05/+0.01422 | L8/0.781 |
| 2/0 | 0.891/+0.0624 | +0.05644 | +0.005971 | +6.216e-05/+0.01227 | L8/0.797 |
| 2/1 | 0.867/+0.07097 | +0.06948 | +0.001656 | +7.293e-05/+0.01607 | L8/0.795 |

## Interpretation boundary

This localizes which earlier writes compose the contextual value available at Layer 9. It does not yet decompose an earlier attention write into its own source positions or QK/value factors. Integrated contributions are path-dependent allocations along the straight original-to-counterfactual state segment; the exact telescope and finite suffix interventions are path-independent endpoints.
