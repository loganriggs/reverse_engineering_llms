# Noisy joint interventions: coverage and conditioning both matter

Completed the predeclared CPU-only known-interface experiment in NOISY_JOINT_DESIGN_PLAN.md: 20 seeds, two noise levels, four budgets, four strategies, 640 fitted endpoints. No native-model work or h128 dependencies changed.

Median noise-free prediction MSE on unseen three/four-contrast interventions at measurement noise SD 0.05:

| Measurements | Random joint | Pair-balanced joint | Designed joint | Singleton |
|---|---:|---:|---:|---:|
| 14 | 0.1323 | 0.1589 | 0.1101 | 0.2162 |
| 28 | 0.1071 | 0.1208 | 0.00210 | 0.2162 |
| 56 | 0.02921 | 0.00353 | 0.00133 | 0.2162 |
| 112 | 0.00218 | 0.00134 | 0.000764 | 0.2162 |

At 28 measurements, designed probes beat both joint baselines in all20 seeds at both noise levels. Random probes have median rank18 of28; balanced and designed probes both have rank28. The full-rank balanced baseline can nevertheless perform poorly: random angles sometimes give a very small product, strongly amplifying noise under unregularized least squares. At noise SD0.2 and budget28 its median error is1.933, versus0.03356 for designed probes and0.2162 for the zero-prediction singleton baseline. At budget112 and either noise level, design beats balanced in17/20 seeds; there is no universal per-instance dominance claim.

## Why this happens

Each acquisition row observes one coefficient theta_ij multiplied by c=d_i*d_j. The information matrix is diagonal with entries I_ij=sum(c^2) across measurements of that pair. Singleton probes give I=0; missing pairs give zero diagonal entries. A nonzero but tiny entry identifies a coefficient noiselessly while producing high uncertainty with noise. For an observed coefficient, conditional on the design, least-squares variance is sigma^2/I_ij. The coefficient prior is used only to generate teachers, not to regularize fitting.

At fixed unit perturbation norm, |d_i*d_j|<=1/2. The greedy determinant criterion favors previously weakly measured pairs and stronger products. Its geometry-based choice uses no labels or held-out test data. Balanced sampling addresses coverage but does not control product strength. This is an elementary linear-design explanation, not a new general theorem or evidence of adaptive semantic discovery.

The test support sizes3/4 never occur in acquisition (size2): recovery composes pairwise effects into unseen combinations under the assumed quadratic family. This does not test unseen higher-order interactions. The duplicate pairs and quadratic hypothesis class are supplied, and a known ordinary output has been subtracted. An unrestricted white-box analyst could inspect the constructed weights directly. This benchmark isolates measurement design under limited observation access; it does not yet test an interpreter with weights, unknown feature coordinates or unknown graph structure.

## Verification and next discriminating test

Independent symmetric-matrix forward versus monomial features: maximum error5.55e-16. Coordinate-wise scalar solves versus numerical least squares:2.28e-13. Checked fixed norms, singleton blindness, disjoint acquisition/test support sizes, unique candidate selection, all640 records and source hashes. A separate physical duplicate-feature calculation passed100 random cases (audit.json). The audit does not independently regenerate all640 records.

The next comparison should include a predeclared noise-aware shrinkage estimator so the advantage is not exaggerated by unstable unregularized baselines. Then hide the duplicate-pair basis or introduce model misspecification, while preserving whole intervention-family holdouts. This panel establishes a positive control for information gathering; it does not substitute for the broader matched-behavior mechanism benchmark or the pending scaling results.
