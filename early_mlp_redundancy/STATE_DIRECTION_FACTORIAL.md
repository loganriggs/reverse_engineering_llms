# M63: faithful responses to supplied directions do not establish faithful task-change representations

The fixed factorial is complete: all 18 final M61 featurizers, 14 pre-existing source-group holdout pairs, both endpoints, and source/final/identity interventions. No models were fitted or selected. These are familiar development families, not new tasks or pristine whole-task transfer.

## What was separated

A **background** is the recipient MLP1 write on which an intervention acts. A **direction** is a donor-minus-recipient write at the previously specified positions. We crossed native versus reconstructed backgrounds with native versus reconstructed directions. Each response is measured relative to its own intact background. The native-background/native-direction response is the common teacher.

The previous test supplied the exact native direction externally. It tested whether a reconstructed state still supported the correct downstream response. The two new corners ask whether the representation itself supplies an adequate task-changing direction. Donor and recipient are encoded independently with their own gates; no norm matching or locked support is used.

Response error is native-probability-weighted squared error in the final log-probability change, divided by teacher response energy. Zero response scores 1. Family means average the six family-specific ratios equally; pooled energy weighting answers a different question. Very weak effects can have large relative errors, so the complete raw energies and margin errors remain in the JSON. Finite nonlinear interactions mean these corners are not an additive attribution decomposition.

## Results

Balanced versus raw causal supervision improves source-position family-mean error in all 6 architecture/seed comparisons when given a native direction, but only 2/6 when using reconstructed directions in native backgrounds. End-to-end reconstructed backgrounds and directions improve 4/6. At final positions the corresponding counts are 5/6, 5/6, and 4/6; relative improvement still leaves every model worse than zero response in both reconstructed-direction corners.

All 18 models beat zero response for source-position reconstructed directions in native backgrounds. Balanced scalar/block models also beat zero end-to-end, but the balanced shared models remain at 1.327 and 1.631. Thus the result is a substantial limitation, not a claim that the representations carry no useful changes.

The table reports equal-family response errors. “Supplied” = reconstructed background/native direction; “encoded” = native background/reconstructed direction; “both” = both reconstructed. Local delta error is pooled squared write-difference error, which has a different denominator and aggregation.

| Architecture | Seed | Objective | Site | Supplied | Encoded | Both | Local delta error |
|---|---:|---|---|---:|---:|---:|---:|
| scalar | 71 | intact_KL | source | 1.434 | 0.688 | 0.932 | 0.880 |
| scalar | 71 | intact_KL | final | 0.852 | 2.200 | 2.824 | 1.501 |
| scalar | 71 | KL_raw_response | source | 1.041 | 0.668 | 0.805 | 0.877 |
| scalar | 71 | KL_raw_response | final | 0.782 | 2.117 | 2.620 | 1.476 |
| scalar | 71 | KL_balanced_response | source | 0.740 | 0.751 | 0.765 | 0.884 |
| scalar | 71 | KL_balanced_response | final | 0.805 | 2.554 | 2.452 | 1.575 |
| block | 71 | intact_KL | source | 1.400 | 0.886 | 0.977 | 0.924 |
| block | 71 | intact_KL | final | 1.051 | 3.093 | 3.426 | 1.570 |
| block | 71 | KL_raw_response | source | 1.219 | 0.886 | 0.906 | 0.926 |
| block | 71 | KL_raw_response | final | 0.944 | 2.566 | 3.022 | 1.478 |
| block | 71 | KL_balanced_response | source | 0.934 | 0.923 | 0.869 | 0.919 |
| block | 71 | KL_balanced_response | final | 0.844 | 2.374 | 3.144 | 1.631 |
| shared_fixed | 71 | intact_KL | source | 1.520 | 0.920 | 1.729 | 0.932 |
| shared_fixed | 71 | intact_KL | final | 1.142 | 1.770 | 1.821 | 1.351 |
| shared_fixed | 71 | KL_raw_response | source | 1.082 | 0.864 | 1.378 | 0.923 |
| shared_fixed | 71 | KL_raw_response | final | 1.201 | 2.120 | 2.245 | 1.393 |
| shared_fixed | 71 | KL_balanced_response | source | 0.888 | 0.896 | 1.327 | 0.926 |
| shared_fixed | 71 | KL_balanced_response | final | 1.058 | 2.033 | 2.772 | 1.377 |
| scalar | 72 | intact_KL | source | 1.135 | 0.817 | 0.908 | 0.885 |
| scalar | 72 | intact_KL | final | 0.991 | 1.705 | 2.872 | 1.320 |
| scalar | 72 | KL_raw_response | source | 0.972 | 0.850 | 0.926 | 0.888 |
| scalar | 72 | KL_raw_response | final | 1.146 | 2.452 | 3.152 | 1.451 |
| scalar | 72 | KL_balanced_response | source | 0.891 | 0.847 | 0.999 | 0.889 |
| scalar | 72 | KL_balanced_response | final | 1.057 | 1.985 | 2.647 | 1.449 |
| block | 72 | intact_KL | source | 2.005 | 0.869 | 1.093 | 0.925 |
| block | 72 | intact_KL | final | 1.263 | 1.533 | 2.015 | 1.465 |
| block | 72 | KL_raw_response | source | 1.682 | 0.855 | 1.014 | 0.924 |
| block | 72 | KL_raw_response | final | 1.256 | 1.799 | 2.717 | 1.680 |
| block | 72 | KL_balanced_response | source | 1.236 | 0.822 | 0.894 | 0.916 |
| block | 72 | KL_balanced_response | final | 1.126 | 1.185 | 2.031 | 1.457 |
| shared_fixed | 72 | intact_KL | source | 1.700 | 0.826 | 1.833 | 0.950 |
| shared_fixed | 72 | intact_KL | final | 1.320 | 1.909 | 2.335 | 1.389 |
| shared_fixed | 72 | KL_raw_response | source | 1.193 | 0.801 | 1.475 | 0.946 |
| shared_fixed | 72 | KL_raw_response | final | 1.574 | 2.394 | 2.638 | 1.105 |
| shared_fixed | 72 | KL_balanced_response | source | 0.938 | 0.834 | 1.631 | 0.945 |
| shared_fixed | 72 | KL_balanced_response | final | 1.238 | 2.027 | 2.503 | 1.396 |

## Conceptual consequences

This experiment distinguishes **preserving a response to an externally provided change** from **representing the change itself**. Reconstruction and response-preserving training do not yet give us reliable computational units. A shared dictionary is architectural reuse, not evidence of native redundancy. These tests also do not identify native signed cancellation or prove that any layer implements clustering.

One falsifiable explanation is that changes in selected supports introduce inaccurate directions. Another is inaccurate coordinates within otherwise appropriate supports; a third is decoder geometry poorly aligned with downstream sensitivity. Correct discrete clustering could also require support changes, so gate switching alone is not evidence of a defect. The next fixed diagnostic separates continuous coordinate changes from support changes and tests their native responses. It is a test of candidate representation boundaries, not a new training sweep.

## Verification and stopping rule

3556 native forwards; all 1512 previous-corner response/margin results replay exactly. The independent NumPy audit checked 4536 full-vocabulary response vectors, 1512 local geometry rows, and 504 reconstructions from saved selected codes. Maximum decoder relative error was 1.02e-7; maximum absolute metric error was 7.45e-9. Decoder checks do not independently validate gate selection. All 14 sample-file hashes and comparison source/result hashes were checked when this report was assembled.

The factorial is closed. No candidate, task, amplitude, loss weight, or seed search follows from its outcomes. The next bounded protocol is `validation/support_change_diagnostic_protocol.json`. Historical whole-task training exclusions remain intact.
