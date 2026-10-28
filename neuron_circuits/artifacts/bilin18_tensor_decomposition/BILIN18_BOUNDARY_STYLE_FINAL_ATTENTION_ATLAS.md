# Layer-17 head/source atlas for the spelling edit

Every changed Layer-17 attention edge is grouped by native head and its source relation to the spelling cue and boundary. A 32-point exact output-gradient line integral allocates the finite spelling contrast over those edge groups, carried residual, final MLP, and the measured float32 attention residual.

## Locked results

- **A0_integrated_component_atlas_recomposes: supported.**
- **A1_known_L17H2_boundary_is_recovered: supported.**
- **A2_nonboundary_attention_is_not_one_component: not supported.**
- **A3_nonboundary_attention_is_broad: not supported.**
- **A4_top_nonboundary_relation_replicates: supported.**
- **A5_distributed_final_attention_ontology_is_supported: not supported.**

| Fold/split | H2-boundary rank | top nonboundary component/share | top-4 share | relation means |
|---|---:|---:|---:|---:|
| 0/0 | 2 | h2_after_boundary / 0.888 | 0.996 | before_cue:-0.000337, cue:-4.06e-05, between:-0.00089, boundary:+0.00572, after_boundary:+0.013, destination:+0.0015 |
| 0/1 | 2 | h2_after_boundary / 0.894 | 0.998 | before_cue:-0.000306, cue:-2.05e-05, between:-0.000945, boundary:+0.00632, after_boundary:+0.0151, destination:+0.00164 |
| 1/0 | 2 | h2_after_boundary / 0.892 | 0.996 | before_cue:-0.000184, cue:-1.53e-05, between:-0.00132, boundary:+0.00694, after_boundary:+0.0163, destination:+0.0018 |
| 1/1 | 2 | h2_after_boundary / 0.874 | 0.997 | before_cue:-0.000476, cue:-2.46e-05, between:-0.000918, boundary:+0.00537, after_boundary:+0.0106, destination:+0.00137 |
| 2/0 | 2 | h2_after_boundary / 0.882 | 0.998 | before_cue:-0.000199, cue:-1.23e-05, between:-0.000932, boundary:+0.00783, after_boundary:+0.014, destination:+0.00177 |
| 2/1 | 2 | h2_after_boundary / 0.888 | 0.998 | before_cue:-0.000406, cue:-2.42e-05, between:-0.00031, boundary:+0.00685, after_boundary:+0.0161, destination:+0.00191 |

## Interpretation boundary

The line-integral allocation is exact for the declared spelling contrast and native finite state segment. It is an effect decomposition, not a claim that each head/relation group independently stores a feature. A broad positive nonboundary support indicates many small terminal reads or query-induced edge changes; a sparse result identifies a shorter typed consumer candidate for finite freezing.
