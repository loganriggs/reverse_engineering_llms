# Top-1 versus runner-up channel decomposition of the dense margin path

The model-selected margin is decomposed exactly into its top-1 logit path minus its runner-up logit path. Each individual channel bank and their frozen union are selected on FineWeb discovery and evaluated OOD.

## Locked hypotheses

- **R0_margin_equals_top1_minus_runnerup: supported.**
- **R1_top1_logit_path_is_native_sparse: supported.**
- **R2_runnerup_logit_path_is_native_sparse: supported.**
- **R3_margin_density_is_created_by_combining_logit_roles: supported.**
- **R4_union_of_logit_banks_compresses_margin: not supported.**
- **R5_logit_banks_are_distinct: not supported.**
- **R6_margin_cancellation_exceeds_individual_logits: supported.**

Maximum margin recomposition error: `3.20e-06`. Top-128 bank overlap: `107` channels.

| Corpus | Top1 K32/K64/K128/K256/K512 (K90) | Runner K32/K64/K128/K256/K512 (K90) | Margin direct K128/K512/K2048/K4096 (K90) | Margin union 16/32/64/128/256/512 per logit | Median mass top1/runner/margin |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.898/0.945/0.955/0.957/0.962 (64) | 0.865/0.909/0.922/0.926/0.927 (64) | 0.656/0.703/0.837/0.957 (4096) | 0.391/0.534/0.620/0.654/0.682/0.736 | 2.03/2.27/5.59 |
| wikitext | 0.903/0.965/0.973/0.975/0.976 (32) | 0.887/0.950/0.962/0.965/0.968 (64) | 0.647/0.695/0.805/0.933 (4096) | 0.309/0.482/0.598/0.667/0.704/0.741 | 2.14/2.39/5.71 |
| tinystories | 0.923/0.970/0.980/0.983/0.985 (32) | 0.927/0.948/0.959/0.969/0.972 (32) | 0.865/0.908/0.949/0.983 (512) | 0.457/0.626/0.759/0.830/0.857/0.883 | 1.95/2.04/5.00 |

If individual logits are sparse but their union is not sufficient, margin density comes from context-varying signed interaction rather than merely combining two fixed sparse output banks.
