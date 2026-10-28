

## H317 — Balanced mixed-head supervision at fixed per-readout reference count

Fit three shared interaction-kernel ridge readouts: actual9-head, angle9-head, mixed18-group supervision. Each uses exactly3072 reference sites across the same768 discovery documents, one selected group per site. Seeded balanced group assignments are shuffled independently of responses; counts differ by at most1. Same feature map, ridge.01, mean-diagonal normalization and float64 Cholesky. Reuse H315 query panel as a diagnostic set. No validation selection or ridge sweep. Unlike H315, a model has one readout for all target groups, not18 specialists.

| Target family | Training mixture | Fidelity | Paired document95% |
|---|---|---:|---:|
|actual|actual|3.94%|[3.03,4.72]%|
|actual|angle|-7.40%|[-8.75,-6.23]%|
|actual|mixed|-1.79%|[-2.99,-0.69]%|
|angle|actual|-2.14%|[-3.35,-1.05]%|
|angle|angle|-2.04%|[-3.53,-0.69]%|
|angle|mixed|-2.22%|[-3.55,-1.02]%|

Locked gates: mixed10: FAIL, mixed80retention: FAIL. Mixed10 requires mixed-readout fidelity>=10% in both target families. Retention requires>=80% of the corresponding H315 specialist-system fidelity in both. The specialist system uses18× as many group/site labels and18 readouts; that retention gate is an efficiency target, not an equal-total-data causal comparison.

Mixed supervision yields−1.79% actual and−2.22% angle fidelity, worse than zero missing-response prediction. Actual-only shared fitting reaches3.94% on actual targets; angle-only reaches−2.04% on angle targets. Per-head results reveal large head6 errors (mixed actual−22.39%, mixed angle−6.70%) despite small positive fidelity on many other heads. Both locked gates fail. Because ridge.01 was inherited from specialists and each group now has only~171 mixed examples, regularization/coverage is unresolved. First audit a locked ridge grid on discovery calibration/evaluation before expanding data or inferring intrinsic specialization.

**What is controlled.** Each shared readout has the same3072 examples and3,538,944 dual output coefficients as ONE specialist. The actual-only and angle-only mixtures distribute examples over9 heads; mixed distributes them over18 groups. Target matrices and input feature representation are unchanged. Since equal budget spreads supervision more thinly, weak performance can reflect sample complexity rather than inability to represent a shared rule. Ridge was inherited, not separately optimized for mixture distributions. One assignment seed gives no estimate of assignment variability.

Reference descriptors add approximately7.67M scalars per readout if all four coordinate arrays are stored without exploiting redundancy (3072×2×(96+1152)), on top of3.54M coefficients. This is still not a compact replacement of the native MLP. The actual independent complement dimension is992, but descriptors are stored in ambient1152 coordinates. Kernel-training and prediction cost also remain substantial. No intact substitution, semantic labeling or downstream behavior evaluation.

**Audit.** Every model uses all3072 reference-site indices once with a balanced group assignment. Source/readout/reference-response hashes checked before use. All three ridge solve residuals<1e-8. Independent aggregation of saved document errors replays summary fidelities within1e-12. Query panel, layer17 and seed explicitly retained from H315/H316. Bootstrap2000 whole-document resamples conditions on fitted readouts and assignments. All jobs terminal; original native weights unchanged.

**Next discriminator.** Vary balanced mixed supervision size while preserving an explicit shared representation and measure the marginal gains per stored coefficient. Compare with reduced-budget specialists to separate coverage from sample count. If a shared kernel improves with more coverage, seek a primal low-rank tensor/readout factorization rather than indefinitely growing dual references. If it does not, diagnose geometry/normalization and estimation before inferring head-specific primitives. Fresh confirmation is needed for any promoted shared predictor. Broad research goal active.
