## H957 — Downstream metric recovers behavior unavailable to Euclidean projection

Tested whether frozen moderate-penalty top-eight output directions can reproduce teacher output changes using coefficients chosen for a downstream metric. Both seeds, all168 factorial swaps, isolated MLP and joint residual paths retained. No weight training. For each recipient, central finite differences at1% and0.5% of teacher effect norm estimate output-logit Jacobians. Center logits/Jacobians under the exact target distribution, then solve weighted least squares with a small ridge and explicit coefficient norm budgets1x and2x the teacher MLP-effect norm. Orthonormal basis coordinates make these output-edit norm budgets. All candidates are replayed through the nonlinear model.

This is a teacher-informed per-example oracle. It estimates the capacity of a frozen span under this procedure; it is neither a learned predictor nor a globally optimized KL bound. Both finite-difference scales, both budgets, and all failures are retained. Euclidean projection, actual top-eight response, zero MLP change, and exact teacher intervention provide controls.

Joint-path KLratio to zero MLP change (lower better), ranges across seeds:

| Template / changed factor | Actual top8 | Euclidean oracle | Fisher oracle, budget1 | Fisher oracle, budget2 |
|---|---|---|---|---|
| prepositional:both | 0.849–0.875 | 0.771–0.940 | 0.351–0.498 | 0.351–0.499 |
| prepositional:distractor | 0.719–0.753 | 0.645–0.834 | 0.295–0.403 | 0.295–0.404 |
| prepositional:subject | 0.979–0.998 | 0.908–0.983 | 0.393–0.548 | 0.391–0.546 |
| relative_clause:both | 0.855–0.883 | 0.801–0.840 | 0.387–0.400 | 0.386–0.400 |
| relative_clause:distractor | 0.858–0.914 | 0.899–0.945 | 0.469–0.490 | 0.469–0.488 |
| relative_clause:subject | 0.688–0.773 | 0.579–0.670 | 0.253–0.296 | 0.252–0.296 |

The Fisher coefficients substantially improve output fidelity without changing the span. Across cells, budget1 joint KLratios are.253–.548, versus actual.688–.998. This is a concrete counterexample to concluding behavioral span insufficiency from H956's large Euclidean residual alone. The same directions can reproduce more of the downstream distribution than their learned or Euclidean coefficients achieved.

The gain is incomplete and metric-specific. Prepositional subject-only task error remains46–51% even while joint KLratio improves to.393–.548; relative-clause subject-only task error improves to4.6–7.4%. There is no universal ordering of task-margin and distribution fidelity. Absolute KL remains small on this panel, as documented in H955.

Doubling the norm budget gives little further improvement. Maximum observed edit norm is1.354x teacher effect norm, well within budget2. This is not evidence of on-manifold validity: teacher-informed coefficients can compensate for missing computations through different directions. The oracle does not explain how a featurizer would predict those coefficients or whether the same representation works on unseen inputs.

Numerics: finite-difference step-halving discrepancies have median0.755% and maximum1.734% in reference-weighted Jacobian norm. Every candidate receives nonlinear validation, so reported KLs are actual model values, not quadratic predictions. Solver KKT residual2.71e-15; norm constraints satisfied; nine isotropic closed-form solver checks agree within1.43e-14. Basis orthogonality1.34e-15. Identity logits2.77e-5; exact historical joint margins8.04e-6. Actual/Euclidean comparator replay versus H955/H956 differs by at most1.09e-5 in margin and1.29e-7 KL. Independent summary reaggregation passes. 25,536/25,536 whole-model forwards, no new teacher endpoint data, no weight training, unchanged inputs, reserved test untouched, natural document pool26,112.

Interpretation: the user's distinction among activation faithfulness and output-distribution equivalence is experimentally consequential. A compact span can be much more adequate under a downstream metric than Euclidean projection suggests. However a different coefficient vector per directed intervention may describe a collection of local compensations rather than a reusable feature representation.

Next: test coefficient integrability on the complete factorial graph. A fixed additive representation requires c(A→B)=z(B)-z(A), so reversals must negate and every closed loop must sum to zero. Quantify oracle violations against exact/Euclidean controls, project the oracle edge coefficients onto consistent state potentials, and replay those changes downstream. This separates the expressive gain from direction-by-direction freedom from a representation that can assign one feature state to each activation. It is still a teacher-informed diagnostic; transfer learning should follow only after this consistency check.
