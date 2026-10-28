# L8 natural-state decoder plus source-specific response correction

No fitting on new panel. Qhybrid=Qnatural+P(Q-Qnatural)+(Q-Qnatural)P-P(Q-Qnatural)P. Source projectors use c_proj weights only. Base rank 2r plus source rank r requires at most 4r quadratic reads plus one exact linear read per scalar. Wrong-head control rotates each declared head identity by +1 modulo nine. These are source-indexed charts, not a recovered global forward router.

Locked gates: `{'hybrid128_response': True, 'hybrid128_natural': False, 'hybrid256_response': True, 'hybrid256_natural': True, 'source_identity': True, 'pair_generalization': True, 'exact': True}`.

| Method | Natural R2 | Fit heads | Held-out heads | Half | Pairs |
|---|---:|---:|---:|---:|---:|
| natural64 | 0.843 | 0.515 | 0.530 | 0.497 | 0.517 |
| natural128 | 0.910 | 0.630 | 0.676 | 0.644 | 0.654 |
| natural256 | 0.960 | 0.801 | 0.835 | 0.818 | 0.821 |
| exact | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid128_correct | 0.855 | 0.788 | 0.803 | 0.805 | 0.774 |
| hybrid128_wrong | 0.855 | 0.546 | 0.556 | 0.526 | 0.704 |
| hybrid256_correct | 0.922 | 0.917 | 0.942 | 0.943 | 0.912 |
| hybrid256_wrong | 0.922 | 0.680 | 0.716 | 0.690 | 0.859 |

Natural R2 for hybrid methods is averaged over nine source charts. The source identity is supplied by the intervention declaration, not inferred from model state. This experiment asks whether known source relations shorten a causal description; it does not establish a single executable forward feature circuit. Wrong-source assignment is one fixed control, not a permutation significance test.

## Post-result row bootstrap

These 95% intervals resample 128 complete text rows (2,000 replicates), retaining all sites, heads and coordinates within each row. They quantify this panel’s row-sampling uncertainty, not model-selection uncertainty or corpus-wide guarantees.

| Budget | Held-out response 95% interval | Advantage vs equal-budget global | Advantage vs wrong source |
|---|---|---|---|
| 128 | [0.767, 0.834] | [0.081, 0.176] | [0.186, 0.313] |
| 256 | [0.925, 0.955] | [0.074, 0.142] | [0.175, 0.280] |
