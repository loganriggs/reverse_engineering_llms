# Native channels inside the essential final-MLP backward path

Each final-layer bilinear channel's exact backward contribution is contracted into the existing 32D Layer-12 behavioral tangent. Shared and endpoint-specific marginal rankings are frozen on FineWeb discovery and evaluated on disjoint FineWeb, WikiText, and TinyStories.

## Locked hypotheses

- **N0_native_channels_recompose_final_mlp_path: supported.**
- **N1_shared_512_channel_bank_is_sufficient: not supported.**
- **N2_shared_2048_channel_bank_is_sufficient: not supported.**
- **N3_endpoint_specific_512_banks_are_sufficient: not supported.**
- **N4_individual_paths_are_native_dense: not supported.**
- **N5_native_path_uses_signed_cancellation: not supported.**
- **N6_shared_bank_beats_random: supported.**
- **N7_shared_native_bank_transfers_ood: supported.**

Maximum all-channel behavioral recomposition error: `1.13e-06`.

| Corpus | Component | Shared K32/K128/K512/K1024/K2048/K4096 | Shared K90 | Specific K90 | Random512 p95 | Median norm support | Median projection support | Median absolute projection mass |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fineweb | prediction | 0.902/0.964/0.971/0.979/0.984/0.994 | 32 | 32 | 0.349 | 134 | 49 | 1.84 |
| fineweb | target | 0.875/0.947/0.959/0.969/0.978/0.992 | 64 | 64 | 0.329 | 300 | 82 | 2.31 |
| fineweb | margin | 0.543/0.664/0.697/0.754/0.836/0.961 | 4096 | 4096 | 0.277 | 1164 | 408 | 5.59 |
| wikitext | prediction | 0.915/0.977/0.983/0.986/0.989/0.995 | 32 | 32 | 0.393 | 144 | 50 | 1.93 |
| wikitext | target | 0.898/0.965/0.971/0.975/0.981/0.994 | 64 | 64 | 0.384 | 375 | 103 | 2.73 |
| wikitext | margin | 0.498/0.672/0.725/0.768/0.834/0.956 | 4096 | 4096 | 0.256 | 1166 | 392 | 5.71 |
| tinystories | prediction | 0.943/0.986/0.991/0.993/0.994/0.996 | 32 | 32 | 0.313 | 114 | 43 | 1.77 |
| tinystories | target | 0.928/0.982/0.988/0.990/0.992/0.995 | 32 | 32 | 0.305 | 196 | 62 | 2.07 |
| tinystories | margin | 0.655/0.848/0.902/0.925/0.950/0.980 | 512 | 512 | 0.348 | 851 | 313 | 5.00 |

Marginal rankings are stable fixed-bank tests, not globally optimal subset solutions. Broad native support would establish dense execution in the trained CP basis while leaving open a compact symmetry-aware tensor refactorization of the same quadratic form.
