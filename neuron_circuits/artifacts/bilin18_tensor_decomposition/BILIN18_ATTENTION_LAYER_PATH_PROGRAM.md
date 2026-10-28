# Exact two-QK attention layer paths in the 546M model

The manual derivative includes both normalized rotary QK factors, their unnormalized product pattern, current-value response, the fixed persistent Layer-0 value bus, OV projection, RMSNorms, and every intervening bilinear MLP. Each of Layers 13–17 contributes either baseline transport `I` or its complete attention response `A`, giving 32 exact layer-subset paths. The top-four MLP program and attention ordering are frozen on separate FineWeb panels.

## Locked hypotheses

- **T0_manual_two_qk_attention_jvp_is_exact: supported.**
- **T1_attention_layer_paths_recompose_exactly: supported.**
- **T2_full_analytic_jvp_predicts_finite_effect: supported.**
- **T3_mlp_identity_path_explains_most_effect: supported.**
- **T4_four_attention_paths_are_high_fidelity: supported.**
- **T5_compact_combined_program_predicts_native_effect: supported.**
- **T6_attention_paths_beat_random: not supported.**
- **T7_attention_program_transfers_ood: supported.**
- **T8_attention_program_is_low_order: supported.**
- **T9_context_specific_attention_selection_is_small: supported.**

Maximum native-forward/JVP error: `2.74e-07`. Maximum path recomposition error: `9.33e-07`.
Top-four paths with attention degree ≤1: **4/4**.

Frozen order: `IIIII`, `IIIIA`, `AIIII`, `IAIII`, `IIAII`, `AIIIA`, `IIIAI`, `AAIII`, `AIIAI`, `IAIIA`, `IAIAI`, `IIAIA`, `AIAII`, `AAIIA`, `IIIAA`, `AIIAA`, `IAIAA`, `IAAII`, `AAIAI`, `IIAAI`, `AIAAI`, `AIAIA`, `AAAII`, `IAAAI`, `IIAAA`, `AAIAA`, `IAAIA`, `AAAIA`, `AIAAA`, `AAAAI`, `IAAAA`, `AAAAA`.

| Corpus | Identity | K4 analytic | K4 oracle/random-p95 | Full analytic→finite | MLP4 + K4 attention→finite |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.976 | 0.998 | 0.998/0.991 | 1.000 | 0.954 |
| wikitext | 0.972 | 0.997 | 0.997/0.988 | 1.000 | 0.947 |
| tinystories | 0.970 | 0.997 | 0.998/0.992 | 1.000 | 0.962 |

An `A` label means the complete two-QK plus value attention derivative at that layer, followed by the native state-conditioned downstream maps. It is not a head, semantic feature, or fixed matrix. Further QK/value and head decomposition is justified only if this layer-level program is exact and transferably compact.
