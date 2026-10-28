# Centered and factorized late-MLP operator atlas

Each frozen spectral operator mode is converted into an exact weight-derived state read vector and then SVD-factorized between the 32D perturbation input and 1,152D output. Coefficients are never fit or chosen per example. The centered spectrum and frozen mean test whether an uncentered spike is merely a shared operator.

## Locked hypotheses

- **C0_exact_read_vectors_recover_mode_coefficients: supported.**
- **C1_layer16_spike_is_a_common_operator: supported.**
- **C2_layer16_variation_remains_spectrally_spiked: not supported.**
- **C3_layer16_leading_mode_is_a_simple_matrix_circuit: supported.**
- **C4_layer16_mode_family_is_inner_low_rank: supported.**
- **C5_rank4_inner_factorization_preserves_the_atlas: not supported.**
- **C6_layer16_has_a_compact_read_transform_write_program: not supported.**
- **C7_earlier_layers_remain_broad_after_tensor_factorization: supported.**
- **C8_exact_read_program_matches_oracle_projection: supported.**

Maximum exact-read coefficient error: `3.08e-07`.

| Corpus | Layer | Mean | Center top share / R90 | Mode-mean cos | Top/median inner R90 | Full64 | Inner R1 / R2 / R4 / R8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fineweb | 13 | 0.045 | 0.026 / 256 | 0.992 | 19 / 26 | 0.308 | 0.065 / 0.087 / 0.113 / 0.155 |
| fineweb | 14 | 0.025 | 0.023 / 262 | 0.988 | 23 / 26 | 0.267 | 0.038 / 0.056 / 0.079 / 0.118 |
| fineweb | 15 | 0.205 | 0.036 / 251 | 0.999 | 4 / 26 | 0.439 | 0.216 / 0.247 / 0.277 / 0.313 |
| fineweb | 16 | 0.724 | 0.133 / 206 | 1.000 | 3 / 3 | 0.862 | 0.446 / 0.678 / 0.782 / 0.814 |
| wikitext | 13 | 0.043 | 0.026 / 256 | 0.992 | 19 / 26 | 0.314 | 0.069 / 0.092 / 0.119 / 0.160 |
| wikitext | 14 | 0.023 | 0.023 / 262 | 0.988 | 23 / 26 | 0.270 | 0.041 / 0.060 / 0.082 / 0.122 |
| wikitext | 15 | 0.211 | 0.036 / 251 | 0.999 | 4 / 26 | 0.446 | 0.220 / 0.253 / 0.284 / 0.321 |
| wikitext | 16 | 0.756 | 0.133 / 206 | 1.000 | 3 / 3 | 0.877 | 0.471 / 0.695 / 0.800 / 0.831 |
| tinystories | 13 | 0.066 | 0.026 / 256 | 0.992 | 19 / 26 | 0.349 | 0.075 / 0.101 / 0.129 / 0.176 |
| tinystories | 14 | 0.040 | 0.023 / 262 | 0.988 | 23 / 26 | 0.303 | 0.045 / 0.067 / 0.093 / 0.136 |
| tinystories | 15 | 0.229 | 0.036 / 251 | 0.999 | 4 / 26 | 0.484 | 0.233 / 0.267 / 0.302 / 0.343 |
| tinystories | 16 | 0.746 | 0.133 / 206 | 1.000 | 3 / 3 | 0.886 | 0.466 / 0.691 / 0.799 / 0.833 |

A compact triad has three independently testable parts: a state read vector, a low-rank perturbation-to-output transform, and a causal output write. Exact coefficient recovery validates the algebra but does not make the read semantic; token/context selectivity and downstream intervention transfer are separate tests.
