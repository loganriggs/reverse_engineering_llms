# Full-suffix causality of the typed attention-edge vocabulary

Fine typed-edge supports and their broader closures are selected on discovery FineWeb. At 2,048 later held-out token positions per layer, the native local attention write is replaced only at the queried position, that layer's bilinear MLP is recomputed, and the exact remaining transformer produces logits. Recovery is normalized against deleting the complete local attention write.

Maximum native-branch error: `5.34e-05`.

## Locked hypotheses

- **C0_manual_program_matches_native: supported.**
- **C1_typed128_recovers_suffix_behavior: not supported.**
- **C2_typed32_is_behaviorally_useful: supported.**
- **C3_local_operator_recovery_predicts_suffix_recovery: supported.**
- **C4_broader_closures_preserve_behavior: supported.**
- **C5_middle_layers_remain_broader_behaviorally: supported.**

| Layer | Fine32 local/KL | Fine128 local/KL | Drop category KL | Drop distance KL | Drop head KL | Fine32 CE / top1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.529/0.697 | 0.713/0.788 | 0.784 | 0.748 | 0.846 | +0.0150 / 0.949 |
| 1 | 0.691/0.841 | 0.902/0.957 | 0.911 | 0.936 | 0.920 | +0.0534 / 0.899 |
| 2 | 0.458/0.523 | 0.750/0.788 | 0.708 | 0.703 | 0.711 | +0.0134 / 0.943 |
| 3 | 0.434/0.397 | 0.732/0.704 | 0.628 | 0.590 | 0.560 | +0.0110 / 0.947 |
| 4 | 0.506/0.555 | 0.772/0.802 | 0.726 | 0.666 | 0.708 | +0.0128 / 0.927 |
| 5 | 0.732/0.756 | 0.895/0.915 | 0.795 | 0.842 | 0.975 | +0.3239 / 0.788 |

Local-operator versus suffix-KL Spearman correlation across the 30 method/layer cells: `0.901`.

## Guardrails

Only one layer's attention write at one destination is replaced; all other positions and the full suffix are native. KL recovery can be high even when individual top-1 predictions change, and CE delta can be negative on finite samples. Closure methods execute more fine edges than fine32 while using fewer broad predicates, so they optimize description length rather than equal execution. The selected vocabulary is human-declared and does not test whether a learned latent edge ontology would be shorter.
