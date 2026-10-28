# Dynamic-attention boundary of the compact 546M MLP path program

A Layer-12 perturbation is propagated through Layers 13–17 in two finite worlds: suffix attention writes fixed to their native base values, or attention fully recomputed. The frozen top-four I/B program is taken unchanged from the high-count discovery artifact. `Exact attention correction` means the measured full-minus-frozen finite response; it is a scope diagnostic, not yet a compressed attention explanation.

## Locked hypotheses

- **A0_mlp_route_explains_most_native_effect: supported.**
- **A1_attention_correction_is_material: supported.**
- **A2_four_mlp_paths_predict_native_effect: supported.**
- **A3_exact_attention_correction_closes_the_gap: supported.**
- **A4_attention_profile_transfers_ood: supported.**
- **A5_compact_mlp_error_dominates_attention_error: not supported.**

Frozen top four: `IIII`, `IIIB`, `IIBB`, `IIBI`.

| Corpus | Full MLP route→native | Four MLP paths→native | Four + exact attention | Attention projection / norm |
|---|---:|---:|---:|---:|
| fineweb | 0.973 | 0.933 | 0.967 | 0.045 / 0.165 (compact/attention residual 1.09) |
| wikitext | 0.961 | 0.910 | 0.961 | 0.080 / 0.197 (compact/attention residual 1.00) |
| tinystories | 0.970 | 0.928 | 0.968 | 0.059 / 0.173 (compact/attention residual 1.03) |

A small signed projection with a larger norm means attention redirects or cancels the MLP-carried effect rather than simply adding a parallel semantic feature. The failed A5 gate is informative: after adding the exact attention correction, the remaining four-path MLP truncation residual is 1.00--1.09 times the entire omitted-attention correction. The next step is to decompose that correction into symmetry-aware QK and value path families and freeze that compression on separate data.
