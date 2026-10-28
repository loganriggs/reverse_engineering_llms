# Layer-8 construction of the L9H8 boundary style value

Every Layer-8 attention edge into the selected boundary is differenced under the one-token cue swap and partitioned by native head and source relation. The exact edge writes are transported through the Layer-9 RMSNorm/value map, inserted into L9H8's boundary edge with native routing, and propagated through the remaining model.

## Locked results

- **B0_edge_partition_recomposes: supported.**
- **B1_direct_cue_read_is_material: supported.**
- **B2_boundary_self_is_not_the_source: supported.**
- **B3_one_head_is_not_the_whole_builder: not supported.**
- **B4_builder_head_identity_replicates: supported.**
- **B5_direct_two_hop_route_is_supported: supported.**
- **B6_full_attention_component_is_causal: supported.**

| Fold/split | Full win/mean | source means: before / cue / between / self | strongest head share/mean |
|---|---:|---:|---:|
| 0/0 | 0.898/+0.03594 | -0.00231 / +0.04242 / -0.003499 / -0.0006448 | H2 0.991/+0.03559 |
| 0/1 | 0.852/+0.04556 | -0.002585 / +0.05093 / -0.00239 / -0.0003802 | H2 0.990/+0.04519 |
| 1/0 | 0.883/+0.04426 | -0.00247 / +0.05024 / -0.002914 / -0.0005469 | H2 0.994/+0.04398 |
| 1/1 | 0.859/+0.03192 | -0.001816 / +0.03724 / -0.003011 / -0.0004579 | H2 0.993/+0.03164 |
| 2/0 | 0.883/+0.0499 | -0.002226 / +0.0549 / -0.002022 / -0.0007118 | H2 0.993/+0.04955 |
| 2/1 | 0.891/+0.04515 | -0.002471 / +0.05011 / -0.001939 / -0.0005365 | H2 0.982/+0.04438 |

## Interpretation boundary

The source relation describes where the Layer-8 boundary invocation reads after the cue swap; it is not yet a semantic label for the source residual state. A direct cue contribution supports a literal two-hop route. A between-position contribution instead implies that the style signal has already been relayed into other token states. These are mediated contributions to L9H8's declared spelling contrast, not complete functions of the Layer-8 heads.
