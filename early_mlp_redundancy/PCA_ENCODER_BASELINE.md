# M73: variance preservation is not computational fidelity

Two global affine PCA encoders were fitted to exactly 4,092 unique native MLP1 training positions from natural documents 0–31. Ranks 32 and 128 were frozen before fitting; no task labels, causal targets, evaluation documents, rank search or whitening were used. Each write is independently projected around the training mean. This tests the geometry of MLP output writes, not redundancy of native neurons.

## Results

Lower is better for errors and KL. Natural local error here is the mean of 32 document-level squared relative errors; the preflight's pooled errors are 0.5093/0.3466, a different aggregation.

| Measure | PCA32 | PCA128 | Dense learned dictionaries, all 18 |
|---|---:|---:|---:|
| Natural local squared relative error | 0.5197 | 0.3545 | 0.3742–0.3892 |
| Natural output KL | 0.8832 | 0.1350 | 0.0958–0.1162 |
| Natural CE increase | 0.8955 | 0.1362 | See full comparison JSON |
| Natural top-1 agreement | 53.5% | 80.7% | See full comparison JSON |
| Intact task output KL | 0.6687 | 0.1565 | 0.0448–0.1068 |
| Source encoded-change error, native background | 0.8372 | 0.6140 | 0.5481–0.6499 |
| Final encoded-change error, native background | 0.8265 | 0.8687 | 0.7204–0.9263 |
| Source error, reconstructed background and change | 1.6125 | 0.7789 | 0.5017–0.6591 |
| Final error, reconstructed background and change | 1.7244 | 1.7944 | 0.9058–1.3968 |

Causal rows use an equal average over six task families. Error 1 is predicting no response, not a faithfulness threshold. Pooled and answer-margin results are retained separately: for example, PCA128 final encoded-change pooled error is 0.5708 despite family error 0.8687. Rank128 is not universally better than rank32 on family averages. Both PCA models retain high binary task accuracy (27/28 and 28/28), despite substantial output-distribution changes.

PCA128 reconstructs natural writes better than every dense dictionary projection but has worse natural and intact-task KL than every one. Both reconstructed causal corners are also worse than all18 dense projections. The native-background encoded-change comparison is mixed (PCA128 better than 8/18 source and 3/18 final). The18 references share data and training ancestry; these are descriptive comparisons, not independent replications.

## Conceptual interpretation

Retaining high-variance directions does not suffice to preserve computation. At the same 128-dimensional output rank, orientation of the retained subspace matters. The learned dictionaries have training histories that include downstream objectives; this comparison does not isolate which training ingredient caused their advantage. It also does not establish an advantage for conditional groups: the dense dictionaries and PCA are both global affine maps.

A local theoretical account is downstream sensitivity. If e is a discarded write residual, J is the downstream logit Jacobian and F=diag(p)-pp^T, then KL(p(w)||p(w+alpha e)) is approximately alpha² e^T J^T F J e / 2 near alpha=0. PCA minimizes Euclidean residual energy, not this behavioral quadratic form. A smaller Euclidean error can therefore produce larger KL. Finite replacement also permits nonlinear effects; M73 does not distinguish those explanations.

This concerns distributed representation and downstream use. It is not proof of literal removal redundancy, cancellation between native channels, semantic manifolds, or reusable computational units. The prior native cancellation/substitutability distinctions still apply. Rank32's coordinate budget is not equal model-class capacity to conditional32-of128 encoders. PCA storage includes mean plus basis: 38,016 or148,608 float64 scalars, with no claim of matched training parameter count or inference speed.

## Verification and reproducibility

- fit_pca_encoder.py verifies the frozen protocol's reference/cohort/cache hashes, train/eval document disjointness, and saves exact indices, mean, covariance, spectrum and basis. Training variance explained: 43.36%/66.77%. Relative cutoff gaps: 0.03556/0.004339.
- audit_pca_encoder.py independently reconstructs selection, computes chunked Torch moments and direct rectangular-data SVD. Covariance relative difference 2.72e-16; projector disagreement at most 6.95e-13. All4,096 heldout natural positions pass nesting and cast checks. Fit state SHA is recorded before downstream scoring.
- probe_pca_factorial.py: 420 native forwards, 14 familiar pairs, both endpoints, source/final/identity and all three measured background/change corners. All28 native baselines exactly replay; teacher effects reuse hash-bound native records. No dictionary controls refitted or rerun.
- audit_pca_factorial.py: all504 saved full-vocabulary response vectors,168 geometry rows and56 independently reconstructed PCA writes checked. Maximum metric absolute difference7.45e-9 (including large local energies).
- evaluate_pca_natural.py:96 forwards over32 heldout documents, exact native-write replay. All64 candidate/document full outputs saved and independently rescored by audit_pca_natural.py; maximum metric discrepancy2.67e-15.
- analyze_pca_evaluation.py preserves every raw, pooled, family, margin, local, intact and natural result; all M72 references are hash-bound. No experimental failure or omitted condition occurred in M73.

The fit NPZ and full response tensors remain in the local artifact directory; the public packet contains scripts, protocols, receipts, results and hashes, not activation banks. All whole-task training exclusions persist. This is the familiar development panel, not a new-task discovery test.

## Decision and bounded next step

Close the PCA rank comparison: no further rank sweep. Prefer the existing dense learned spans as the stronger current behavioral baseline, without claiming semantic features or adequacy. Before another architecture/objective change, test the downstream-sensitivity explanation against finite nonlinear disruption using fixed residual directions and a local Taylor prediction. Its deliverable is a decision about the metric a featurizer should preserve, not another ranking of reconstruction scores. See downstream_residual_metric_protocol.json. Return to joint featurizer/manifold validation after this bounded diagnostic.
