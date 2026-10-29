# M71: coefficient and support limits matter; full-span projection suggests a concrete baseline

All18 fixed dictionaries contain directions that yield better causal predictions than their current encoders produce. **Union-span projection improves equal-family response error18/18 at both sites. Full-dictionary projection improves further18/18 at source positions and16/18 at final positions.** All full-dictionary scores beat zero response, whereas none of the actual encoders did at final positions.

This identifies useful directions and coefficients already available inside the frozen dictionaries. It does not prove sufficient capacity for accurate computation: full-span final errors still range about0.720–0.926, where zero response is1. Beating that weak baseline is far from a complete mechanistic explanation.

## What the diagnostic does

For each native donor-minus-recipient write, orthogonally project into three nested spaces: recipient-selected atoms, the union of donor/recipient selected atoms, and all128 dictionary atoms. Compare these against the actual encoded difference in the original native recipient background. Native teacher responses, task addresses and model states are fixed. Each selected shared atom appears once.

The projection minimizes local Euclidean error, not downstream response error. Thus the local nesting inequalities are guaranteed mathematically and are implementation checks; the downstream comparisons are empirical. A failed full-span projection would not prove that the span contains no better downstream direction.

| Comparison, equal-family response error | Source positions | Final positions |
|---|---:|---:|
| Recipient projection beats actual encoding | 10/18 | 18/18 |
| Union projection beats actual encoding | 18/18 | 18/18 |
| Full dictionary beats union projection | 18/18 | 16/18 |
| Full dictionary beats actual encoding | 18/18 | 18/18 |
| Recipient projection beats zero response | 18/18 | 7/18 |
| Union projection beats zero response | 18/18 | 8/18 |
| Full dictionary beats zero response | 18/18 | 18/18 |

The improvements are metric-dependent. Union versus actual improves pooled response error13/18 at source and17/18 at final, and pooled answer-margin error6/18 and15/18. Full dictionary versus union improves pooled response error18/18 at both sites, but answer-margin error18/18 source and15/18 final. All family/mask-free raw comparisons remain in the JSON; no candidate is selected.

## A useful correction: the full-span case need not be an oracle

The protocol initially described the projections collectively as teacher-informed diagnostics. That is necessary for the conditional recipient/union projections as tested, but **too broad for the fixed full dictionary**. If P_D projects onto the full dictionary span, then

`P_D(w_d - w_r) = P_D w_d - P_D w_r`.

With the frozen mean mu, scale s and decoder bias b, define a per-input least-squares encoder

`a(w) = D^+ [(w - mu)/s - b]`,

`f(w) = mu + s [D a(w) + b]`.

Its difference is exactly the full-span projected difference in real arithmetic. Each endpoint is encoded independently, without task labels, donor information or a teacher response. We verified this on all504 endpoint reconstructions and1512 differences; maximum relative squared discrepancy against the applied M71 full-span directions was1.05e-15. Small discrepancies reflect float32 rounding in native differences and applied vectors.

This gives a concrete, deployable **dense least-squares baseline** using the same dictionary. It generally uses128 coordinates instead of at most32 selected coordinates, so it is not a sparsity-matched improvement. Nor have we yet tested its independently reconstructed backgrounds or natural-text retention. The algebra establishes direction equivalence, not complete intervention equivalence after numerical reconstruction or downstream deployment.

A second deployable comparison is to solve least squares on each input's own original selected support. This keeps the support budget but changes coefficients. It is not the same as projecting a pair's difference into their union; that distinction is precisely what the next test will resolve.

## Detailed equal-family errors

| Architecture | Seed | Objective | Site | Actual | Recipient | Union | Full dictionary |
|---|---:|---|---|---:|---:|---:|---:|
| scalar | 71 | intact_KL | source | 0.688 | 0.840 | 0.658 | 0.557 |
| scalar | 71 | intact_KL | final | 2.200 | 1.063 | 1.110 | 0.803 |
| scalar | 71 | KL_balanced_response | source | 0.751 | 0.855 | 0.685 | 0.555 |
| scalar | 71 | KL_balanced_response | final | 2.554 | 1.068 | 1.105 | 0.812 |
| block | 71 | intact_KL | source | 0.886 | 0.911 | 0.875 | 0.580 |
| block | 71 | intact_KL | final | 3.093 | 0.966 | 0.965 | 0.817 |
| block | 71 | KL_balanced_response | source | 0.923 | 0.937 | 0.899 | 0.583 |
| block | 71 | KL_balanced_response | final | 2.374 | 0.945 | 0.981 | 0.821 |
| shared_fixed | 71 | intact_KL | source | 0.920 | 0.824 | 0.747 | 0.601 |
| shared_fixed | 71 | intact_KL | final | 1.770 | 1.079 | 1.003 | 0.856 |
| shared_fixed | 71 | KL_balanced_response | source | 0.896 | 0.829 | 0.750 | 0.602 |
| shared_fixed | 71 | KL_balanced_response | final | 2.033 | 1.033 | 1.051 | 0.856 |
| scalar | 72 | intact_KL | source | 0.817 | 0.871 | 0.724 | 0.648 |
| scalar | 72 | intact_KL | final | 1.705 | 0.902 | 0.887 | 0.920 |
| scalar | 72 | KL_balanced_response | source | 0.847 | 0.872 | 0.762 | 0.649 |
| scalar | 72 | KL_balanced_response | final | 1.985 | 0.957 | 0.955 | 0.926 |
| block | 72 | intact_KL | source | 0.869 | 0.803 | 0.720 | 0.650 |
| block | 72 | intact_KL | final | 1.533 | 1.095 | 1.155 | 0.839 |
| block | 72 | KL_balanced_response | source | 0.822 | 0.792 | 0.713 | 0.648 |
| block | 72 | KL_balanced_response | final | 1.185 | 1.034 | 1.079 | 0.840 |
| shared_fixed | 72 | intact_KL | source | 0.826 | 0.802 | 0.758 | 0.623 |
| shared_fixed | 72 | intact_KL | final | 1.909 | 1.173 | 1.165 | 0.769 |
| shared_fixed | 72 | KL_balanced_response | source | 0.834 | 0.806 | 0.765 | 0.626 |
| shared_fixed | 72 | KL_balanced_response | final | 2.027 | 1.221 | 1.214 | 0.757 |
| scalar | 71 | KL_encoded_balanced_response | source | 0.722 | 0.840 | 0.670 | 0.548 |
| scalar | 71 | KL_encoded_balanced_response | final | 1.645 | 1.034 | 1.054 | 0.786 |
| block | 71 | KL_encoded_balanced_response | source | 1.005 | 0.918 | 0.879 | 0.581 |
| block | 71 | KL_encoded_balanced_response | final | 1.252 | 0.886 | 0.899 | 0.783 |
| shared_fixed | 71 | KL_encoded_balanced_response | source | 0.938 | 0.838 | 0.749 | 0.603 |
| shared_fixed | 71 | KL_encoded_balanced_response | final | 1.623 | 1.032 | 0.972 | 0.800 |
| scalar | 72 | KL_encoded_balanced_response | source | 0.805 | 0.892 | 0.760 | 0.629 |
| scalar | 72 | KL_encoded_balanced_response | final | 1.445 | 0.812 | 0.755 | 0.894 |
| block | 72 | KL_encoded_balanced_response | source | 0.957 | 0.783 | 0.714 | 0.647 |
| block | 72 | KL_encoded_balanced_response | final | 1.464 | 0.943 | 0.917 | 0.730 |
| shared_fixed | 72 | KL_encoded_balanced_response | source | 0.809 | 0.805 | 0.759 | 0.611 |
| shared_fixed | 72 | KL_encoded_balanced_response | final | 2.052 | 1.132 | 1.112 | 0.720 |

## Verification and scope

All504 support/coordinate checks independently reproduce the saved selections with positive boundary gaps. Local projection orthogonality and idempotence errors are below3e-15. The actual encoded difference lies in the union span up to float32 rounding (maximum relative residual energy1.51e-11). Full≤union≤recipient and union≤actual local error inequalities pass within recorded numerical tolerances.

4060 native forwards completed. All1512 actual-encoding controls replay M70 response and margin errors exactly. An independent QR-then-small-SVD calculation verifies all1512 projected cases, ranks and local energies; all6048 final response vectors pass NumPy scoring. Maximum applied-projection relative squared discrepancy is7.58e-16, maximum response-metric absolute discrepancy1.25e-14. No job is still running.

All14 familiar source-group evaluation pairs and18 models were retained. No new data, fit, coefficient tuning, amplitude tuning or candidate selection occurred. The next protocol compares independent per-input own-support and full-dictionary least-squares encoders through the complete causal factorial and natural retention. Native redundancy, meaningful manifolds and new-task discovery remain separate unfulfilled claims.
