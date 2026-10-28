# Exact backward path program through the final suffix

The final output covector is propagated through four exact path types: final residual versus final bilinear MLP, crossed with direct mixed-state versus attention response. These are evaluated as context-executed weight programs, not fitted static maps.

## Locked hypotheses

- **P0_four_path_backward_program_is_exact: not supported.**
- **P1_two_backward_paths_are_sufficient: not supported.**
- **P2_three_backward_paths_are_high_fidelity: supported.**
- **P3_attention_response_is_material: not supported.**
- **P4_final_mlp_response_is_material: supported.**
- **P5_path_order_transfers_ood: supported.**
- **P6_short_program_can_have_broad_path_vectors: not supported.**
- **P7_context_executed_program_resolves_static_map_failure: not supported.**

Maximum write-adjoint recomposition error: `2.38e-04`; same-graph internal error: `1.15e-07`; behavioral recomposition error: `1.29e-06`.

| Path count | Discovery-selected paths | Discovery aggregate fidelity |
|---:|---|---:|
| 1 | final_residual__attention | 0.103 |
| 2 | final_residual__mixed_direct, final_mlp__mixed_direct | 0.958 |
| 3 | final_residual__mixed_direct, final_residual__attention, final_mlp__mixed_direct | 0.977 |

| Corpus | Component | K1/K2/K3/K4 | No attention | No final MLP | Static fitted map |
|---|---|---:|---:|---:|---:|
| fineweb | ce | 0.083/0.985/0.985/1.000 | 0.985 | -0.052 | 0.236 |
| fineweb | prediction | 0.142/0.947/0.959/1.000 | 0.947 | -0.782 | 0.450 |
| fineweb | target | 0.113/0.958/0.963/1.000 | 0.958 | -0.720 | 0.369 |
| fineweb | margin | -0.007/0.991/0.992/1.000 | 0.991 | 0.361 | 0.254 |
| wikitext | ce | 0.022/0.984/0.990/1.000 | 0.984 | 0.389 | 0.238 |
| wikitext | prediction | 0.163/0.978/0.983/1.000 | 0.978 | -0.502 | 0.423 |
| wikitext | target | 0.141/0.980/0.985/1.000 | 0.980 | -0.363 | 0.380 |
| wikitext | margin | -0.018/0.989/0.990/1.000 | 0.989 | 0.525 | 0.190 |
| tinystories | ce | -0.097/0.972/0.974/1.000 | 0.972 | -0.927 | -0.177 |
| tinystories | prediction | 0.204/0.901/0.984/1.000 | 0.901 | -2.456 | 0.539 |
| tinystories | target | 0.214/0.897/0.983/1.000 | 0.897 | -3.157 | 0.373 |
| tinystories | margin | -0.101/0.980/0.957/1.000 | 0.980 | -0.152 | 0.034 |

| Component | final_residual__mixed_direct | final_residual__attention | final_mlp__mixed_direct | final_mlp__attention |
|---|---:|---:|---:|---:|
| ce | 94 | 39 | 68 | 28 |
| prediction | 51 | 17 | 19 | 21 |
| target | 99 | 26 | 40 | 23 |
| margin | 96 | 37 | 70 | 32 |

A low path count is algebraic program length. The R90 values show how broad each path's realized adjoint family remains in residual coordinates. Thus a short exact program and a broad activation/readout atlas can coexist.
