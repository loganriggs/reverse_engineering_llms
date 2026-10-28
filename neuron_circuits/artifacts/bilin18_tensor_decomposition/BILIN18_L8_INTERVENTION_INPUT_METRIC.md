# L8 intervention-aware input metrics

Four scalar output programs; every state decoder uses at most 64 quadratic reads plus one exact linear read per scalar. The role secant is only a response diagnostic.

Selected mixture weight: 1.0. Locked gates: `{'exact': True, 'heldout_heads': False, 'heldout_families': False, 'natural_retention': False, 'role_function': False}`.

| Method | Natural R2 | Fit-head removal | Held-out heads | Half removal | Paired removal |
|---|---:|---:|---:|---:|---:|
| natural64 | 0.823 | 0.522 | 0.507 | 0.481 | 0.514 |
| mixed64 | 0.769 | 0.641 | 0.451 | 0.483 | 0.486 |
| raw64 | 0.740 | 0.479 | 0.536 | 0.519 | 0.528 |
| symmetric_role32 | 0.601 | 0.458 | 0.281 | 0.311 | 0.315 |
| random64_0 | 0.033 | 0.042 | 0.025 | 0.031 | 0.022 |
| random64_1 | 0.030 | 0.043 | 0.016 | 0.023 | 0.018 |
| random64_2 | 0.031 | 0.040 | 0.014 | 0.021 | 0.014 |
| random64_3 | 0.028 | 0.047 | 0.020 | 0.027 | 0.019 |
| role_secant32 | 0.601 | 0.469 | 0.288 | 0.317 | 0.317 |
| exact | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Unsymmetrized role-secant triangle-cycle relative RMS: 0.012. Nonzero cycles rule out interpreting that response rule as the differences of a single state function. Its natural R2 column evaluates the symmetrized scalar form, not an independently valid secant-based state decoder.

The mixture weight was selected on discovery rows using even-head removal only, then frozen before test collection. Odd heads, half removals, and paired removals are held-out perturbations. This is local source-response fidelity for four output directions, not whole-layer reconstruction or semantic identification.
