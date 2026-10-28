# Prospective finite-edit path compiler through all 18 layers

One shared checkpoint-native prefix JVP produces pre-MLP tangents for all 18 layers on the same examples. Midpoint-selected actual channel deltas and realized-local controls use the same rank 256 and are propagated through each complete suffix. Five fit nodes and four interleaved held-out nodes measure per-example product-path degree.

## Locked hypotheses

- **D0_small_edit_tangent_is_valid: supported.**
- **D1_midpoint_direction_survives_full_depth: supported.**
- **D2_midpoint_support_survives_full_depth: supported.**
- **D3_midpoint_causal_fidelity_survives_full_depth: supported.**
- **D4_quadratic_path_survives_full_depth: supported.**
- **D5_path_remains_nearly_planar_full_depth: supported.**
- **D6_a_path_compiler_break_exists_after_layer4: not supported.**
- **D7_layer04_result_replicates: supported.**

First registered threshold failure by corpus: `{"fineweb": {"midpoint_cosine": null, "midpoint_fidelity": null, "quadratic_path_fidelity": null, "rank2_path_energy": null, "support_jaccard": null}, "tinystories": {"midpoint_cosine": null, "midpoint_fidelity": null, "quadratic_path_fidelity": null, "rank2_path_energy": null, "support_jaccard": null}}`.

| Layer | Corpus | Small/base/mid cosine | Mid norm error | Support Jaccard | Mid/realized-local fidelity (ratio) | Path d1/d2/d3 | Rank-2 energy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.994/0.410/0.994 | 0.383 | 0.842 | 0.551/0.580 (0.951) | 0.968/0.996/1.000 | 0.9990 |
| 0 | tinystories | 0.996/0.531/0.996 | 0.350 | 0.869 | 0.648/0.710 (0.913) | 0.974/0.997/1.000 | 0.9993 |
| 1 | fineweb | 0.996/0.592/0.975 | 0.359 | 0.695 | 0.737/0.762 (0.968) | 0.968/0.997/1.000 | 0.9960 |
| 1 | tinystories | 0.995/0.496/0.980 | 0.369 | 0.724 | 0.733/0.734 (0.999) | 0.969/0.996/0.999 | 0.9964 |
| 2 | fineweb | 0.996/0.663/0.972 | 0.390 | 0.646 | 0.720/0.759 (0.948) | 0.969/0.997/1.000 | 0.9952 |
| 2 | tinystories | 0.995/0.498/0.975 | 0.389 | 0.679 | 0.752/0.792 (0.950) | 0.967/0.996/0.999 | 0.9963 |
| 3 | fineweb | 0.996/0.626/0.971 | 0.332 | 0.690 | 0.718/0.773 (0.928) | 0.969/0.996/0.999 | 0.9956 |
| 3 | tinystories | 0.995/0.530/0.974 | 0.321 | 0.695 | 0.777/0.804 (0.966) | 0.964/0.996/0.999 | 0.9967 |
| 4 | fineweb | 0.995/0.564/0.970 | 0.341 | 0.684 | 0.678/0.735 (0.922) | 0.968/0.996/0.999 | 0.9961 |
| 4 | tinystories | 0.995/0.547/0.966 | 0.418 | 0.631 | 0.706/0.683 (1.034) | 0.959/0.995/0.999 | 0.9953 |
| 5 | fineweb | 0.994/0.530/0.962 | 0.415 | 0.600 | 0.655/0.627 (1.044) | 0.965/0.996/0.999 | 0.9954 |
| 5 | tinystories | 0.995/0.353/0.964 | 0.535 | 0.580 | 0.601/0.732 (0.821) | 0.955/0.994/0.999 | 0.9938 |
| 6 | fineweb | 0.994/0.534/0.965 | 0.390 | 0.610 | 0.690/0.694 (0.994) | 0.964/0.996/0.999 | 0.9958 |
| 6 | tinystories | 0.995/0.451/0.957 | 0.555 | 0.610 | 0.658/0.704 (0.935) | 0.952/0.994/0.999 | 0.9948 |
| 7 | fineweb | 0.995/0.493/0.959 | 0.393 | 0.668 | 0.664/0.660 (1.007) | 0.965/0.995/0.999 | 0.9958 |
| 7 | tinystories | 0.996/0.407/0.956 | 0.528 | 0.585 | 0.637/0.697 (0.914) | 0.960/0.993/0.998 | 0.9937 |
| 8 | fineweb | 0.996/0.535/0.958 | 0.378 | 0.662 | 0.683/0.757 (0.902) | 0.964/0.995/0.999 | 0.9963 |
| 8 | tinystories | 0.996/0.458/0.960 | 0.526 | 0.600 | 0.579/0.578 (1.002) | 0.954/0.994/0.998 | 0.9948 |
| 9 | fineweb | 0.996/0.528/0.965 | 0.406 | 0.631 | 0.643/0.671 (0.957) | 0.962/0.996/0.999 | 0.9966 |
| 9 | tinystories | 0.996/0.450/0.955 | 0.499 | 0.615 | 0.698/0.728 (0.959) | 0.954/0.992/0.997 | 0.9951 |
| 10 | fineweb | 0.996/0.549/0.960 | 0.393 | 0.620 | 0.648/0.725 (0.894) | 0.968/0.995/0.999 | 0.9962 |
| 10 | tinystories | 0.996/0.438/0.954 | 0.491 | 0.605 | 0.693/0.708 (0.979) | 0.950/0.991/0.997 | 0.9950 |
| 11 | fineweb | 0.995/0.572/0.963 | 0.370 | 0.636 | 0.732/0.775 (0.945) | 0.971/0.995/0.999 | 0.9965 |
| 11 | tinystories | 0.996/0.436/0.950 | 0.489 | 0.580 | 0.510/0.595 (0.856) | 0.945/0.991/0.997 | 0.9935 |
| 12 | fineweb | 0.996/0.577/0.959 | 0.361 | 0.673 | 0.622/0.694 (0.895) | 0.971/0.995/0.998 | 0.9963 |
| 12 | tinystories | 0.997/0.455/0.944 | 0.486 | 0.615 | 0.633/0.682 (0.928) | 0.949/0.991/0.996 | 0.9940 |
| 13 | fineweb | 0.996/0.573/0.959 | 0.364 | 0.652 | 0.656/0.671 (0.978) | 0.974/0.995/0.999 | 0.9967 |
| 13 | tinystories | 0.996/0.425/0.949 | 0.513 | 0.585 | 0.601/0.670 (0.896) | 0.948/0.991/0.997 | 0.9942 |
| 14 | fineweb | 0.996/0.591/0.967 | 0.367 | 0.684 | 0.586/0.599 (0.978) | 0.973/0.996/0.999 | 0.9966 |
| 14 | tinystories | 0.996/0.424/0.949 | 0.466 | 0.580 | 0.609/0.636 (0.958) | 0.953/0.991/0.996 | 0.9940 |
| 15 | fineweb | 0.996/0.572/0.964 | 0.369 | 0.646 | 0.619/0.682 (0.908) | 0.973/0.996/0.999 | 0.9966 |
| 15 | tinystories | 0.996/0.433/0.950 | 0.496 | 0.590 | 0.534/0.643 (0.830) | 0.952/0.991/0.996 | 0.9945 |
| 16 | fineweb | 0.996/0.643/0.963 | 0.342 | 0.730 | 0.581/0.604 (0.962) | 0.975/0.996/0.999 | 0.9972 |
| 16 | tinystories | 0.996/0.419/0.959 | 0.449 | 0.600 | 0.583/0.612 (0.953) | 0.947/0.992/0.996 | 0.9951 |
| 17 | fineweb | 0.996/0.556/0.980 | 0.308 | 0.736 | 0.763/0.710 (1.074) | 0.978/0.996/0.998 | 0.9980 |
| 17 | tinystories | 0.995/0.494/0.962 | 0.413 | 0.657 | 0.687/0.722 (0.951) | 0.940/0.993/0.996 | 0.9965 |

The sweep locates a depth transition only if direction, support, causal retention, or held-out path geometry degrades under an unchanged assay. Passing through Layer 4 would mean the low-degree conditional-program ontology extends beyond the directly lexical layers.
