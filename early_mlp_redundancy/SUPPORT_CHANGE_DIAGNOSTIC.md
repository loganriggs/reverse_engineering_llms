# M64: feature switching helps source changes but hurts small final-position changes

The fixed diagnostic is complete for all 18 M61 models and 14 familiar source-group holdout pairs. Native interventions use MLP1, the second block. No fitting, family filtering, or candidate selection occurred.

## Question and decomposition

Let a_r and a_d be raw encoder coordinates for recipient and donor writes, q_r and q_d their independently selected supports, and D the unit-column dictionary. With global scale omitted below, the reconstructed difference is

`D[(a_d - a_r) * q_r] + D[a_d * (q_d - q_r)]`.

The first term changes coordinates while retaining the recipient's support (**within**). The second changes which atoms contribute (**switch**). Their sum is the full reconstructed difference. Shared atoms use the union of selected groups and contribute only once. Decoder bias and centering cancel. Both terms were independently applied in the native recipient background, compared against the same native donor-minus-recipient teacher response. This is a diagnostic intervention, not a replacement encoder or a new fitted model.

## Results

| Site | Within beats full | Switch beats full | Within beats zero response | Switch beats zero response |
|---|---:|---:|---:|---:|
| Changed input tokens | 4/18 | 0/18 | 18/18 | 14/18 |
| Final position | 18/18 | 14/18 | 9/18 | 0/18 |

Counts use equal-family means of native-probability-weighted squared final log-probability response errors. Zero response scores 1. All full-direction final-position scores exceed 1. The final-position within-only scores improve in every model but only half become better than zero. At source positions, the complete change usually outperforms either component alone. Thus a blanket claim that switching is harmful is contradicted by the source results.

Local and downstream measures remain different: final-position within-only local squared error is below 1 for every model, but downstream response error is below 1 for only nine. The downstream reader weights directions unevenly.

| Architecture | Seed | Objective | Site | Full response error | Within error | Switch error | Within/native energy | Switch/native energy |
|---|---:|---|---|---:|---:|---:|---:|---:|
| scalar | 71 | intact_KL | source | 0.688 | 0.887 | 0.764 | 0.140 | 0.163 |
| scalar | 71 | intact_KL | final | 2.200 | 0.914 | 2.075 | 0.115 | 0.684 |
| scalar | 71 | KL_raw_response | source | 0.668 | 0.879 | 0.760 | 0.142 | 0.165 |
| scalar | 71 | KL_raw_response | final | 2.117 | 0.917 | 1.883 | 0.119 | 0.656 |
| scalar | 71 | KL_balanced_response | source | 0.751 | 0.916 | 0.827 | 0.144 | 0.171 |
| scalar | 71 | KL_balanced_response | final | 2.554 | 0.925 | 2.329 | 0.116 | 0.779 |
| block | 71 | intact_KL | source | 0.886 | 0.910 | 0.987 | 0.143 | 0.119 |
| block | 71 | intact_KL | final | 3.093 | 0.992 | 2.903 | 0.120 | 0.662 |
| block | 71 | KL_raw_response | source | 0.886 | 0.955 | 0.937 | 0.145 | 0.110 |
| block | 71 | KL_raw_response | final | 2.566 | 0.996 | 2.326 | 0.121 | 0.567 |
| block | 71 | KL_balanced_response | source | 0.923 | 0.977 | 0.962 | 0.146 | 0.099 |
| block | 71 | KL_balanced_response | final | 2.374 | 0.992 | 2.196 | 0.122 | 0.722 |
| shared_fixed | 71 | intact_KL | source | 0.920 | 0.870 | 1.091 | 0.145 | 0.096 |
| shared_fixed | 71 | intact_KL | final | 1.770 | 1.303 | 1.621 | 0.107 | 0.396 |
| shared_fixed | 71 | KL_raw_response | source | 0.864 | 0.869 | 1.020 | 0.145 | 0.087 |
| shared_fixed | 71 | KL_raw_response | final | 2.120 | 1.303 | 1.702 | 0.108 | 0.436 |
| shared_fixed | 71 | KL_balanced_response | source | 0.896 | 0.878 | 1.043 | 0.146 | 0.091 |
| shared_fixed | 71 | KL_balanced_response | final | 2.033 | 1.308 | 1.692 | 0.111 | 0.440 |
| scalar | 72 | intact_KL | source | 0.817 | 0.887 | 0.959 | 0.149 | 0.180 |
| scalar | 72 | intact_KL | final | 1.705 | 0.846 | 1.775 | 0.111 | 0.440 |
| scalar | 72 | KL_raw_response | source | 0.850 | 0.931 | 0.985 | 0.152 | 0.181 |
| scalar | 72 | KL_raw_response | final | 2.452 | 0.894 | 2.120 | 0.108 | 0.601 |
| scalar | 72 | KL_balanced_response | source | 0.847 | 0.875 | 0.984 | 0.151 | 0.185 |
| scalar | 72 | KL_balanced_response | final | 1.985 | 0.917 | 1.723 | 0.113 | 0.580 |
| block | 72 | intact_KL | source | 0.869 | 0.891 | 0.987 | 0.137 | 0.124 |
| block | 72 | intact_KL | final | 1.533 | 1.176 | 1.851 | 0.116 | 0.588 |
| block | 72 | KL_raw_response | source | 0.855 | 0.891 | 0.976 | 0.138 | 0.125 |
| block | 72 | KL_raw_response | final | 1.799 | 1.213 | 2.000 | 0.115 | 0.802 |
| block | 72 | KL_balanced_response | source | 0.822 | 0.884 | 0.940 | 0.144 | 0.110 |
| block | 72 | KL_balanced_response | final | 1.185 | 1.156 | 1.484 | 0.114 | 0.592 |
| shared_fixed | 72 | intact_KL | source | 0.826 | 0.805 | 0.993 | 0.183 | 0.127 |
| shared_fixed | 72 | intact_KL | final | 1.909 | 1.225 | 1.657 | 0.112 | 0.469 |
| shared_fixed | 72 | KL_raw_response | source | 0.801 | 0.803 | 0.968 | 0.192 | 0.124 |
| shared_fixed | 72 | KL_raw_response | final | 2.394 | 1.200 | 1.782 | 0.116 | 0.173 |
| shared_fixed | 72 | KL_balanced_response | source | 0.834 | 0.795 | 1.007 | 0.195 | 0.116 |
| shared_fixed | 72 | KL_balanced_response | final | 2.027 | 1.306 | 1.699 | 0.107 | 0.474 |

## Theoretical interpretation and limits

A hard TopK representation is piecewise affine in its input. On a scalar selection boundary, swapping selected atom i for j changes the decoded contribution by `a_j D_j - a_i D_i`. The coordinates can have nonzero magnitude at the boundary, so an arbitrarily small input change can produce a finite decoded jump unless the relevant contributions match. Block and shared-group boundaries have the same issue, with the jump restricted to atoms entering or leaving the union. This is a property of the candidate representation, not a discovery of discontinuous native computation.

This mechanism predicts a distinction between large changes that legitimately recruit different features and small changes that accidentally cross a selection boundary. The source/final contrast is consistent with it, but does not establish it: position, native change size, and task effects covary. We have not traced an interpolation through each boundary or proved that specific jumps are spurious.

A major confound remains **attenuation**. Within-only final-position directions have only about 0.107–0.122 times the native change energy, whereas switch components have about 0.173–0.802. Removing switching also shrinks the intervention. A smaller inaccurate direction can beat a larger inaccurate direction, especially for weak teacher responses. The next single fixed control scales the full direction to the within direction's norm using only the candidate vectors, with no teacher-based tuning.

The donor-anchored decomposition was also checked locally: `D[(a_d-a_r)*q_d] + D[a_r*(q_d-q_r)]`. With both reciprocal endpoints and aligned positions, pooled donor/recipient within errors coincide by exchange symmetry; this is not evidence that anchor choice is irrelevant for individual causal responses. Only recipient-anchored components received native intervention tests.

These results concern featurizer boundary fidelity. They do not establish native removal redundancy, native cancellation, semantic clustering, good manifolds, or newly discovered tasks. In particular, a sum of two representation components should not be confused with redundant native channels. All whole-task training exclusions remain unchanged.

## Verification and stopping rule

Independent NumPy encoder and support selection matched all 504 saved encodings with strictly positive selection-boundary gaps. Both decoded decompositions reproduce the actual full direction; maximum relative squared discrepancy is 1.56e-11. The native experiment completed 2758 forwards, exactly replaying all 1512 M63 full-direction controls. The independent audit checked all 4536 final response vectors and 1512 local geometry rows, with maximum response-metric absolute discrepancy 2.14e-14. Identity controls are zero. All sample hashes are bound by the result files.

This diagnostic is closed. `validation/support_norm_control_protocol.json` specifies the one remaining magnitude control. It is not an amplitude sweep or a training change. After that control, consolidate the representation-boundary finding and decide whether to replace hard selection or test continuous within-feature geometry; do not keep optimizing this development panel.
