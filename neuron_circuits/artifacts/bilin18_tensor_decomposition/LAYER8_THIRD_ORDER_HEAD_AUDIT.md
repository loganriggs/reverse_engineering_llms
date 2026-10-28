# Head decomposition of singular third-order suffix paths

Every below-0.8 Mobius failure is paired with the same tensor query and layer triple at the other selected site in its corpus. The third-order response is decomposed exactly by head at the last selected attention layer.

## Locked hypotheses

- **H0_head_decomposition_recomposes: not supported.**
- **H1_singular_residual_is_head_localized: supported.**
- **H2_singular_residual_is_amplified: supported.**
- **H3_singular_residual_is_diffuse: not supported.**

Maximum recomposition error: `9.430e-03`. Median top-head share: `0.923`. Median effective support: `1.17` heads. Median matched amplification: `23.7x`.

| Corpus | Query/row | Triple | Top head/share | Effective support | Relative residual failure/control | Amplification |
|---|---|---|---:|---:|---:|---:|
| fineweb | 3/22 | [9, 11, 13] | 0/0.637 | 2.15 | 2.500e-02/8.219e-03 | 3.0x |
| fineweb | 10/291 | [9, 10, 11] | 6/0.455 | 3.83 | 1.988e-02/1.765e-02 | 1.1x |
| wikitext | 1/271 | [9, 10, 14] | 6/0.858 | 1.34 | 3.095e-01/1.112e-02 | 27.8x |
| wikitext | 1/271 | [9, 11, 14] | 6/0.826 | 1.45 | 8.797e-02/4.972e-03 | 17.7x |
| wikitext | 1/271 | [9, 13, 14] | 6/0.944 | 1.12 | 3.970e-01/1.882e-02 | 21.1x |
| wikitext | 1/271 | [10, 12, 14] | 6/0.923 | 1.17 | 1.450e-01/1.678e-03 | 86.4x |
| wikitext | 1/271 | [10, 12, 17] | 2/0.820 | 1.47 | 1.440e-02/1.596e-03 | 9.0x |
| wikitext | 1/271 | [10, 13, 14] | 6/0.944 | 1.12 | 4.415e-01/8.622e-03 | 51.2x |
| wikitext | 1/271 | [10, 13, 15] | 6/0.982 | 1.04 | 2.310e-01/5.789e-03 | 39.9x |
| wikitext | 1/271 | [10, 14, 15] | 6/0.990 | 1.02 | 4.028e-01/1.586e-02 | 25.4x |
| wikitext | 1/271 | [11, 12, 14] | 6/0.897 | 1.24 | 7.866e-02/1.982e-03 | 39.7x |
| wikitext | 1/271 | [11, 13, 14] | 6/0.939 | 1.13 | 2.418e-01/1.022e-02 | 23.7x |
| wikitext | 1/271 | [11, 14, 15] | 6/0.991 | 1.02 | 1.287e-01/7.006e-03 | 18.4x |

Head localization identifies only the final factor of the ordered path. A diffuse result would motivate subspace rather than head-level decomposition; a concentrated result motivates tracing the dominant head's QK and value factors backward.
