# Semantic atlas of the tangent-bundle output core

The atlas uses 16,000 FineWeb semantic-discovery sites, 32,000 disjoint FineWeb sites, and 12,800 sites from each OOD corpus. Weight-only vocabulary labels, natural target/top1 adjoints, and random rotations of the identical physical subspace are evaluated separately.

## Locked hypotheses

- **S0_large_disjoint_atlas_completed: supported.** The atlas contains 16,000 semantic-discovery sites, 32,000 disjoint FineWeb sites, and 12,800 sites from each OOD corpus.
- **S1_weight_axes_have_lexical_semantics: supported.** Frozen output axes have median held-out vocabulary AUROC at least 0.65 against preregistered token labels.
- **S2_weight_axes_are_more_semantic_than_rotations: not supported.** Native HOSVD axes exceed the p95 median AUROC of independently selected random rotations of the same rank-16 output subspace by at least 0.03.
- **S3_target_adjoint_semantics_transfer: supported.** Frozen per-axis target-adjoint explanations have median AUROC at least 0.60 on FineWeb, WikiText, and TinyStories.
- **S4_top1_adjoint_semantics_transfer: supported.** Frozen per-axis top1-adjoint explanations have median AUROC at least 0.60 on FineWeb, WikiText, and TinyStories.
- **S5_output_variables_dominate_axis_labels: supported.** At least 12 of 16 target-adjoint and 12 of 16 top1-adjoint axes select their corresponding output-token variable rather than current token or position.
- **S6_output_subspace_has_transferable_semantics: supported.** A discovery-frozen linear decoder from the complete rank-16 output subspace has median AUROC at least 0.65 on every corpus for both target and top1 label families.
- **S7_weight_and_natural_roles_agree: not supported.** At least half of axes select the same base token property in weight-only vocabulary geometry and either target- or top1-adjoint semantics.
- **S8_rotation_control_is_matched: supported.** Every rotation control is an orthogonal change of coordinates inside the exact same rank-16 physical output subspace.

## Summary

Weight-only native-axis median vocabulary AUROC: **0.828**; rotation p95 **0.842**; isotropic p95 **0.659**.

| Corpus | target axes | target subspace | top1 axes | top1 subspace | target/top1 rotation p95 |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.905 | 0.945 | 0.892 | 0.954 | 0.946/0.952 |
| wikitext | 0.778 | 0.909 | 0.816 | 0.930 | 0.850/0.878 |
| tinystories | 0.946 | 0.956 | 0.940 | 0.968 | 0.976/0.978 |

Corresponding output-variable selections: target 13/16; top1 12/16. Weight/natural property agreement: 1/16.

## Mode examples

| Mode | weight label / AUROC | target label / FW-Wiki-Tiny | top1 label / FW-Wiki-Tiny | promoted → suppressed |
|---:|---|---|---|---|
| 0 | flag:word_continuation / 0.91 | current:flag:newline / 0.92/0.84/0.96 | current:flag:opening_delimiter / 0.90/0.87/NA | '��' → ' ■' |
| 1 | generic:leading_space / 0.82 | top1:coarse:subword / 0.95/0.93/0.97 | target:coarse:subword / 0.95/0.94/0.97 | 'dylib' → 'result' |
| 2 | flag:non_ascii / 0.83 | current:coarse:capitalized / 0.78/0.77/0.76 | current:coarse:capitalized / 0.80/0.80/0.75 | '=(' → ' Paul' |
| 3 | coarse:content_word / 0.91 | target:flag:newline / 0.85/0.99/0.83 | current:flag:all_caps / 0.83/0.77/0.79 | ' un' → ' IMAGES' |
| 4 | flag:word_continuation / 0.80 | target:coarse:subword / 0.88/0.91/0.94 | top1:coarse:subword / 0.88/0.93/0.94 | 'errors' → ' parser' |
| 5 | coarse:subword / 0.92 | target:flag:colon_semicolon / 0.80/0.78/0.81 | top1:flag:colon_semicolon / 0.82/0.79/0.81 | 'nown' → ' principle' |
| 6 | coarse:conjunction / 0.65 | target:flag:comma_period / 0.86/0.31/0.92 | top1:flag:comma_period / 0.89/0.67/0.93 | ' Identified' → 'AGES' |
| 7 | coarse:auxiliary / 0.74 | target:flag:comma_period / 0.96/0.78/0.98 | top1:flag:comma_period / 0.97/0.92/0.99 | ' forms' → 'akable' |
| 8 | coarse:auxiliary / 0.82 | target:flag:colon_semicolon / 0.84/0.57/0.73 | top1:flag:comma_period / 0.87/0.69/0.89 | ' responds' → 'maxwell' |
| 9 | coarse:determiner / 0.87 | target:flag:comma_period / 0.98/0.77/0.99 | top1:flag:comma_period / 0.98/0.89/0.99 | '.' → 'Adds' |
| 10 | flag:suffix_ing / 0.81 | target:flag:comma_period / 0.96/0.47/0.99 | top1:flag:comma_period / 0.97/0.80/0.99 | ' [...]' → '.' |
| 11 | flag:suffix_ing / 0.86 | target:flag:comma_period / 0.99/0.57/1.00 | top1:flag:comma_period / 0.99/0.83/1.00 | '.' → 'Discussion' |
| 12 | coarse:auxiliary / 0.81 | target:flag:comma_period / 0.94/0.56/0.95 | top1:flag:comma_period / 0.92/0.75/0.94 | ' are' → 'playing' |
| 13 | flag:suffix_ing / 0.87 | target:flag:suffix_ing / 0.83/0.83/0.81 | top1:flag:comma_period / 0.85/0.77/0.90 | ' airs' → ' working' |
| 14 | flag:suffix_ing / 0.89 | target:coarse:subword / 0.89/0.94/0.91 | top1:coarse:subword / 0.89/0.94/0.90 | ' (_' → ' deciding' |
| 15 | coarse:auxiliary / 0.82 | target:flag:comma_period / 0.98/0.59/1.00 | top1:flag:comma_period / 0.99/0.84/1.00 | ' work' → '.}' |

Weight tails describe a direction's potential output effect. Natural adjoint labels describe which output questions read that direction. Agreement is stronger evidence than either alone. Same-subspace rotations determine whether meaning belongs to the reported HOSVD axes or only to their joint span.
