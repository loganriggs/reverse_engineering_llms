# Value-norm and heavy-tail red team for attention source roles

The frozen source-feature edges are tested on another untouched FineWeb split. Controls retain source count and relative-distance bins and now preferentially match the native value-vector norm within each bin. Aggregate, sitewise, 95%-trimmed, and site-bootstrap effects are separated.

Maximum native-branch error: `5.10e-05`.

## Locked hypotheses

- **V0_native_branch_is_exact: supported.**
- **V1_local_privilege_survives_value_matching: supported.**
- **V2_local_privilege_is_bootstrap_robust: supported.**
- **V3_local_privilege_is_not_top_tail_only: supported.**
- **V4_downstream_privilege_survives_value_matching: supported.**
- **V5_broad_sitewise_role_is_supported: not supported.**

| Candidate | Local aggregate / trimmed / CI | Local win / median log2 | KL aggregate / trimmed / CI |
|---|---:|---:|---:|
| L0 H8 `opening_delimiter` | 34.01 / 13.12 / [16.00,93.08] | 0.58 / +1.81 | 86.35 / 8.90 / [15.89,325.30] |
| L1 H6 `non_ascii` | 11.14 / 3.05 / [0.95,49.42] | 0.45 / -0.28 | 3.78 / 2.10 / [0.96,12.12] |
| L1 H5 `modal` | 1.24 / 1.59 / [1.02,1.77] | 0.37 / -0.80 | 2.02 / 1.32 / [0.98,5.10] |
| L2 H0 `opening_delimiter` | 7.35 / 6.84 / [3.50,25.76] | 0.62 / +0.84 | 6.73 / 3.78 / [2.12,53.21] |
| L2 H4 `colon_semicolon` | 8.01 / 5.10 / [2.87,21.13] | 0.80 / +2.86 | 5.95 / 3.66 / [1.27,17.39] |
| L3 H7 `numeric` | 1.11 / 0.99 / [0.84,1.67] | 0.34 / -0.73 | 0.96 / 1.04 / [0.40,1.09] |
| L4 H1 `programming_keyword` | 1.76 / 2.06 / [1.18,5.68] | 0.49 / -0.01 | 1.51 / 3.32 / [1.06,5.59] |
| L5 H4 `suffix_s` | 2.16 / 2.17 / [1.71,2.80] | 0.71 / +0.99 | 2.60 / 1.92 / [1.88,3.65] |

## Guardrails

Value-norm matching controls one source of large edge effects but not value direction, exact token identity, or overlapping categories. Bootstrap intervals resample sites and therefore quantify context variability for this panel, not candidate-selection uncertainty. Aggregate and trimmed ratios can support a sparse conditional role even when median log ratio or win rate rejects a broad role; those are intentionally distinct ontologies.
