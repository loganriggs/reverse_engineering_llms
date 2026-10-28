# Prospective finite-causal test of semantic attention-head roles

Six head/feature candidates were frozen after the residual-conditioned probe audit, then evaluated on a third untouched FineWeb split. Each intervention removes one complete native head write at the queried position, recomputes that layer's MLP and the exact remaining transformer, and measures the feature-class logit contrast. Causal sign is fixed on a separate calibration split.

Maximum native-branch versus checkpoint logit error: `8.37e-05`.

## Locked hypotheses

- **C0_manual_native_matches_checkpoint: supported.**
- **C1_probe_roles_have_causal_discrimination: not supported.**
- **C2_selected_heads_beat_other_heads: not supported.**
- **C3_selected_heads_are_top_three_causal: not supported.**
- **C4_attention_role_is_not_only_decodable: not supported.**
- **C5_full_attention_contains_the_role: not supported.**

| Candidate | Signed causal AUROC | Effect size | Rank / 9 | Control median | All-attention AUROC | Mean |class effect| |
|---|---:|---:|---:|---:|---:|---:|
| L0 H8 `numeric` | 0.506 | -0.168 | 5 | 0.520 | 0.542 | 0.01069 |
| L0 H6 `opening_delimiter` | 0.616 | -0.184 | 1 | 0.447 | 0.649 | 0.00918 |
| L1 H7 `negation` | 0.582 | 0.338 | 3 | 0.483 | 0.619 | 0.01539 |
| L1 H6 `non_ascii` | 0.381 | -0.278 | 8 | 0.570 | 0.656 | 0.00824 |
| L2 H1 `dash` | 0.532 | -0.008 | 3 | 0.513 | 0.502 | 0.00532 |
| L3 H4 `numeric` | 0.438 | -0.064 | 9 | 0.484 | 0.552 | 0.01788 |

## Guardrails

The candidates were exploratory on an earlier panel but are prospective on this evaluation split. Signed AUROC tests whether a head's finite causal contribution distinguishes positive from negative contexts; it does not require that the head always promotes the positive class. Removing a head is a large intervention and may expose compensation or nonlinear interactions. Control heads share the same layer and examples but are not matched for write norm. The class contrast averages over the declared vocabulary feature, so rare or heterogeneous labels can still obscure a narrower causal role.
