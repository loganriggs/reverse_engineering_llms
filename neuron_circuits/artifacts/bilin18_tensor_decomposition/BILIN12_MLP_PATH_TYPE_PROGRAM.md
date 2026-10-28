# Executable late-MLP path types in the 12-layer FineWeb tensor model

With attention responses frozen, each of Layers 7–10 has three exact derivative terms: residual (`I`), left-factor bilinear arm (`L`), and right-factor bilinear arm (`R`). Their Cartesian product gives 81 executable path types. A nested path ordering is fit only to discovery FineWeb Jacobians, frozen, and tested on disjoint FineWeb, WikiText, and TinyStories. Random subsets and per-example greedy oracles use the same path inventory.

## Locked hypotheses

- **P0_path_sum_is_exact: supported.**
- **P1_frozen_16_path_program_is_sufficient: supported.**
- **P2_frozen_32_path_program_is_high_fidelity: supported.**
- **P3_frozen_paths_beat_random: not supported.**
- **P4_path_program_transfers_ood: supported.**
- **P5_one_path_is_not_enough: supported.**
- **P6_context_specific_selection_is_material: not supported.**
- **P7_bilinear_path_types_are_material: supported.**
- **P8_program_predicts_finite_effects: supported.**

Maximum independent-JVP recomposition error: `8.51e-07`.
Top-16 paths with bilinear arms in at least two layers: **8/16**.

Top 16 frozen path types: `IIII`, `IIIR`, `RIII`, `IIIL`, `IIRI`, `LIII`, `IRII`, `RRII`, `IILI`, `LLII`, `RIIR`, `IILR`, `LIIL`, `LRII`, `IRIL`, `RILI`.

| Corpus | K | Frozen | Oracle | Random p95 | Finite-effect fidelity |
|---|---:|---:|---:|---:|---:|
| fineweb | 1 | 0.714 | 0.714 | 0.010 | 0.714 |
| fineweb | 4 | 0.829 | 0.837 | 0.108 | 0.829 |
| fineweb | 8 | 0.887 | 0.895 | 0.730 | 0.887 |
| fineweb | 16 | 0.932 | 0.942 | 0.834 | 0.932 |
| fineweb | 32 | 0.975 | 0.980 | 0.892 | 0.975 |
| fineweb | 64 | 0.997 | 0.998 | 0.982 | 0.997 |
| wikitext | 1 | 0.729 | 0.729 | 0.076 | 0.729 |
| wikitext | 4 | 0.848 | 0.853 | 0.735 | 0.848 |
| wikitext | 8 | 0.908 | 0.913 | 0.740 | 0.908 |
| wikitext | 16 | 0.947 | 0.954 | 0.810 | 0.947 |
| wikitext | 32 | 0.983 | 0.986 | 0.900 | 0.983 |
| wikitext | 64 | 0.999 | 0.999 | 0.991 | 0.999 |
| tinystories | 1 | 0.743 | 0.743 | 0.081 | 0.743 |
| tinystories | 4 | 0.856 | 0.861 | 0.657 | 0.856 |
| tinystories | 8 | 0.909 | 0.915 | 0.752 | 0.909 |
| tinystories | 16 | 0.949 | 0.956 | 0.850 | 0.949 |
| tinystories | 32 | 0.984 | 0.987 | 0.923 | 0.984 |
| tinystories | 64 | 0.999 | 0.999 | 0.989 | 0.999 |

A path label names a recurring algebraic operation sequence, not a fixed neuron set or semantic feature. Each path still uses the current text-conditioned RMS Jacobians and bilinear gates. Frozen success would therefore support a reusable operator grammar over contextual operands; oracle-only success would reject that stronger claim.
