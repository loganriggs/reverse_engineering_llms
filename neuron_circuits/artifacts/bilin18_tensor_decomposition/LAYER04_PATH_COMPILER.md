# Prospective finite-edit path compiler through Layers 0--4

One shared checkpoint-native prefix JVP produces pre-MLP tangents for all five layers on the same examples. Midpoint-selected actual channel deltas and realized-local controls use the same rank 256 and are propagated through each complete suffix. Five fit nodes and four interleaved held-out nodes measure per-example product-path degree.

## Locked hypotheses

- **D0_small_edit_tangent_is_valid: supported.**
- **D1_midpoint_direction_survives_to_layer4: supported.**
- **D2_midpoint_support_survives_to_layer4: supported.**
- **D3_midpoint_causal_retention_survives_to_layer4: supported.**
- **D4_quadratic_path_survives_to_layer4: supported.**
- **D5_path_remains_nearly_planar: supported.**
- **D6_depth_introduces_measurable_path_complexity: not supported.**
- **D7_layer01_midpoint_result_replicates: supported.**

| Layer | Corpus | Small/base/mid cosine | Mid norm error | Support Jaccard | Mid/oracle fidelity (retention) | Path d1/d2/d3 | Rank-2 energy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.994/0.409/0.995 | 0.397 | 0.848 | 0.692/0.752 (0.920) | 0.970/0.996/0.999 | 0.9992 |
| 0 | tinystories | 0.995/0.530/0.995 | 0.412 | 0.855 | 0.741/0.759 (0.977) | 0.972/0.996/1.000 | 0.9991 |
| 1 | fineweb | 0.996/0.509/0.974 | 0.385 | 0.684 | 0.623/0.659 (0.946) | 0.963/0.997/1.000 | 0.9959 |
| 1 | tinystories | 0.995/0.583/0.981 | 0.327 | 0.736 | 0.748/0.757 (0.988) | 0.975/0.997/1.000 | 0.9970 |
| 2 | fineweb | 0.995/0.522/0.967 | 0.353 | 0.641 | 0.687/0.787 (0.874) | 0.967/0.997/1.000 | 0.9950 |
| 2 | tinystories | 0.994/0.521/0.979 | 0.332 | 0.712 | 0.793/0.658 (1.205) | 0.970/0.997/0.999 | 0.9966 |
| 3 | fineweb | 0.996/0.499/0.966 | 0.357 | 0.673 | 0.610/0.746 (0.818) | 0.963/0.996/0.999 | 0.9953 |
| 3 | tinystories | 0.994/0.530/0.976 | 0.403 | 0.701 | 0.778/0.791 (0.984) | 0.969/0.996/0.999 | 0.9961 |
| 4 | fineweb | 0.995/0.515/0.971 | 0.327 | 0.646 | 0.725/0.812 (0.893) | 0.967/0.996/0.999 | 0.9961 |
| 4 | tinystories | 0.995/0.579/0.972 | 0.385 | 0.662 | 0.689/0.683 (1.009) | 0.976/0.996/0.999 | 0.9967 |

The sweep locates a depth transition only if direction, support, causal retention, or held-out path geometry degrades under an unchanged assay. Passing through Layer 4 would mean the low-degree conditional-program ontology extends beyond the directly lexical layers.
