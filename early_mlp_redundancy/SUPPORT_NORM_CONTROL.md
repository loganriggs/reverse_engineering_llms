# M65: both magnitude and direction explain the fixed-support advantage

The last control in the support-change series is complete. All 18 fixed featurizers, all 14 existing source-group holdout pairs and both endpoints were retained. These are familiar development examples; no fitting, candidate selection, or new task discovery occurred.

## The control

For each full reconstructed task-change direction d and recipient-support-only direction u, apply `d * ||u|| / ||d||` in the original native recipient background. Norms cover all selected positions and residual coordinates. The scale uses only the two candidate vectors, with no teacher norm, response tuning, clipping or amplitude sweep. Both-zero cases are zero; zero-full/nonzero-within would be explicitly ineligible. No ineligible cases occurred.

The matched vector has the same direction as d and the same local norm as u. This isolates direction at that fixed magnitude; it does not equalize downstream sensitivity or prove an optimal magnitude. All comparisons retain the original native donor-minus-recipient teacher response.

## Results

| Comparison | Source: equal-family | Final: equal-family | Source: pooled | Final: pooled |
|---|---:|---:|---:|---:|
| Matched full beats original full | 4/18 | 18/18 | 0/18 | 17/18 |
| Within-support beats matched full | 10/18 | 13/18 | 0/18 | 15/18 |

At final positions, shrinking the full direction helps every model on equal-family response error. Yet fixed-support directions beat the norm-matched control in 13/18, supporting a direction-specific benefit beyond attenuation in most of these models. It is not universal: all three block-seed72 variants favor matched full on equal-family error, as do two shared-seed71 variants. Within-support is below the zero-response baseline in nine models; matched full only in three. There is still no uniformly faithful computational representation.

At source positions, all matched-full directions beat zero response, and matched full beats within-support in every pooled comparison. The equal-family comparison is mixed (10/18 favor within-support), showing that the answer depends on whether each family or each unit of native response energy gets equal weight. We preserve both instead of selecting a favorable aggregate.

Answer-margin comparisons also differ: within-support beats matched full in 3/18 source and 12/18 final pooled-margin comparisons. Predicting one answer contrast and predicting the whole output response are different requirements.

| Architecture | Seed | Objective | Site | Full | Within | Matched full |
|---|---:|---|---|---:|---:|---:|
| scalar | 71 | intact_KL | source | 0.688 | 0.887 | 0.728 |
| scalar | 71 | intact_KL | final | 2.200 | 0.914 | 1.111 |
| scalar | 71 | KL_raw_response | source | 0.668 | 0.879 | 0.726 |
| scalar | 71 | KL_raw_response | final | 2.117 | 0.917 | 1.111 |
| scalar | 71 | KL_balanced_response | source | 0.751 | 0.916 | 0.781 |
| scalar | 71 | KL_balanced_response | final | 2.554 | 0.925 | 1.179 |
| block | 71 | intact_KL | source | 0.886 | 0.910 | 0.911 |
| block | 71 | intact_KL | final | 3.093 | 0.992 | 1.233 |
| block | 71 | KL_raw_response | source | 0.886 | 0.955 | 0.922 |
| block | 71 | KL_raw_response | final | 2.566 | 0.996 | 1.131 |
| block | 71 | KL_balanced_response | source | 0.923 | 0.977 | 0.947 |
| block | 71 | KL_balanced_response | final | 2.374 | 0.992 | 1.062 |
| shared_fixed | 71 | intact_KL | source | 0.920 | 0.870 | 0.900 |
| shared_fixed | 71 | intact_KL | final | 1.770 | 1.303 | 1.263 |
| shared_fixed | 71 | KL_raw_response | source | 0.864 | 0.869 | 0.871 |
| shared_fixed | 71 | KL_raw_response | final | 2.120 | 1.303 | 1.292 |
| shared_fixed | 71 | KL_balanced_response | source | 0.896 | 0.878 | 0.894 |
| shared_fixed | 71 | KL_balanced_response | final | 2.033 | 1.308 | 1.365 |
| scalar | 72 | intact_KL | source | 0.817 | 0.887 | 0.880 |
| scalar | 72 | intact_KL | final | 1.705 | 0.846 | 1.083 |
| scalar | 72 | KL_raw_response | source | 0.850 | 0.931 | 0.904 |
| scalar | 72 | KL_raw_response | final | 2.452 | 0.894 | 1.113 |
| scalar | 72 | KL_balanced_response | source | 0.847 | 0.875 | 0.891 |
| scalar | 72 | KL_balanced_response | final | 1.985 | 0.917 | 1.083 |
| block | 72 | intact_KL | source | 0.869 | 0.891 | 0.905 |
| block | 72 | intact_KL | final | 1.533 | 1.176 | 0.826 |
| block | 72 | KL_raw_response | source | 0.855 | 0.891 | 0.898 |
| block | 72 | KL_raw_response | final | 1.799 | 1.213 | 0.859 |
| block | 72 | KL_balanced_response | source | 0.822 | 0.884 | 0.875 |
| block | 72 | KL_balanced_response | final | 1.185 | 1.156 | 0.773 |
| shared_fixed | 72 | intact_KL | source | 0.826 | 0.805 | 0.822 |
| shared_fixed | 72 | intact_KL | final | 1.909 | 1.225 | 1.312 |
| shared_fixed | 72 | KL_raw_response | source | 0.801 | 0.803 | 0.808 |
| shared_fixed | 72 | KL_raw_response | final | 2.394 | 1.200 | 1.396 |
| shared_fixed | 72 | KL_balanced_response | source | 0.834 | 0.795 | 0.819 |
| shared_fixed | 72 | KL_balanced_response | final | 2.027 | 1.306 | 1.401 |

All table values are equal-family relative full-distribution response errors; zero response is 1. Raw energies, family rows, pooled and answer-margin scores are retained in `validation/support_norm_comparison.json` and the response files.

## What this changes conceptually

M63 distinguished preserving a reader's response to supplied changes from encoding changes. M64 decomposed encoded changes into within-support movement and feature switching. M65 shows that the final-position failure combines excessive magnitude and, often, less useful direction. That is evidence about the candidate featurizers. It is not evidence that native MLP computation is discrete, redundant, or dominated by cancellation.

Hard selection boundaries can create finite decoded jumps when the selected set changes; that theoretical mechanism remains plausible. We have not proved it explains every observed failure, traced the exact crossing location, or ruled out insufficient dictionary capacity and inaccurate encoder coordinates. No more amplitude or loss tuning on this panel follows.

The useful design requirement is now sharper: a proposed unit should preserve task-relevant *differences*, including small contextual changes, and should survive native causal response tests. Static reconstruction, shared membership, and harmless gate deletion are insufficient. The next experiment returns to native bilinear channel differences, providing continuous, exact targets before proposing another featurizer architecture.

## Verification

2044 native forwards completed. All1512 full-direction results exactly replay M64. Independent NumPy checks verified all1512 matched vectors, zero ineligible cases, and all3024 saved full-vocabulary response vectors. Maximum norm relative error is2.91e-8; maximum metric absolute discrepancy is1.07e-14. The native teacher and original background remain fixed. No independent transformer reimplementation is claimed.

This series is closed. The next protocol is `validation/native_change_coordination_protocol.json`.
