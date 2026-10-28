# Diversity red-team of clean contextual roles

A basis can look clean if every axis has the same dominant output role. This diagnostic therefore pairs within-axis role concentration with between-axis first-role diversity. Effective diversity is the participation rank of the 32 first-role directions; its maximum is 32.

## Locked hypotheses

- **D1_rotation_duplicates_roles: supported.**
- **D2_joint_roles_less_correlated: supported.**
- **D3_diversity_adjusted_cleanliness: supported.**

| Layer | Basis | First-role energy | Effective diversity | Mean pairwise abs cosine | Adjusted cleanliness |
|---|---|---:|---:|---:|---:|
| layer0 | joint | 0.229 | 8.74 | 0.238 | 0.063 |
| layer0 | rotated | 0.305 | 4.28 | 0.390 | 0.041 |
| layer0 | ambient | 0.102 | 15.35 | 0.149 | 0.049 |
| layer17 | joint | 0.715 | 6.51 | 0.290 | 0.146 |
| layer17 | rotated | 0.785 | 4.16 | 0.393 | 0.102 |
| layer17 | ambient | 0.526 | 4.73 | 0.360 | 0.078 |

The adjusted product is a diagnostic, not a canonical objective: it values stable within-axis roles and distinct roles across axes equally. It does not measure completeness or downstream necessity.
