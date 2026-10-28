# CP-2048 recompression of the value–tangent blend

## Full-model behavior

| Model | FineWeb KL | WikiText KL | TinyStories KL |
|---|---:|---:|---:|
| Language CP-2048 | 0.0144 | 0.0189 | 0.0100 |
| Exact rank≤4096 blend | 0.0132 | 0.0178 | 0.0090 |
| Value-recompressed CP-2048 | 0.0138 | 0.0185 | 0.0093 |
| Tangent-recompressed CP-2048 | 0.0137 | 0.0185 | 0.0093 |
| Downstream-recompressed CP-2048 | 0.0137 | 0.0184 | 0.0092 |

## Full-model causal fidelity

| Model | FW Gaussian | FW natural | Wiki Gaussian | Wiki natural | Tiny Gaussian | Tiny natural |
|---|---:|---:|---:|---:|---:|---:|
| Language CP-2048 | 0.173 | 0.628 | 0.203 | 0.633 | 0.099 | 0.617 |
| Exact rank≤4096 blend | 0.185 | 0.644 | 0.223 | 0.649 | 0.130 | 0.623 |
| Value-recompressed CP-2048 | 0.158 | 0.621 | 0.187 | 0.632 | 0.098 | 0.618 |
| Tangent-recompressed CP-2048 | 0.158 | 0.622 | 0.197 | 0.635 | 0.102 | 0.618 |
| Downstream-recompressed CP-2048 | 0.161 | 0.630 | 0.197 | 0.635 | 0.106 | 0.614 |

## Layer-local agreement with the exact blend

| Model | Value | Gaussian tangent | Natural tangent | Isotropic bilinear |
|---|---:|---:|---:|---:|
| Language CP-2048 | 0.9797 | 0.9825 | 0.9804 | 0.9843 |
| Value-recompressed CP-2048 | 0.9993 | 0.9958 | 0.9990 | 0.9931 |
| Tangent-recompressed CP-2048 | 0.9995 | 0.9969 | 0.9993 | 0.9949 |
| Downstream-recompressed CP-2048 | 0.9993 | 0.9967 | 0.9992 | 0.9949 |

## Interpretation

If recompressed models track the exact blend, the apparent rank-4096 construction reflects redundant/aligned terms rather than two independent rank-2048 mechanisms. If value recompression preserves behavior but loses tangent fidelity, value and causal descriptions require different fitting objectives.

This remains an inherited-basis compression result; it does not show independent recovery of the factors.

Raw results: `blend_recompression_results.json`.
