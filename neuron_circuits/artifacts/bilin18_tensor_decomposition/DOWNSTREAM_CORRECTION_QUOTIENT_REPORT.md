# Downstream-weighted correction quotient

## Hypotheses

- **H1_behavior: not supported.**
- **H2_causal: not supported.**
- **H3_transfer: not supported.**
- **H4_beats_pca: supported.**

| Model | FineWeb KL | FW Gaussian | FW natural | Wiki KL | Tiny KL |
|---|---:|---:|---:|---:|---:|
| Learned rank 128 | 0.0139 | 0.160 | 0.630 | 0.0182 | 0.0092 |
| Learned rank 256 | 0.0136 | 0.171 | 0.638 | 0.0180 | 0.0093 |

The learned object is a rank-constrained linear map applied to the exact direct-minus-language correction. It is basis-invariant with respect to CP term order but still evaluates both parent CP models, so it is a causal-description test rather than a runtime compression.

Raw results: `downstream_correction_quotient_results.json`.
