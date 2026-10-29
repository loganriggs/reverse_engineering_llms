# M43: useful repair directions often overshoot at full reconstruction strength

M42 found that predicting a missing native half more accurately can worsen the model’s outputs. M43 tests whether the direction is already harmful near deletion or becomes harmful only at larger amplitude. All six rank32 maps are frozen; no new fits or amplitude optimization.

## Theory made discriminating

For a local illustrative reader metric G, let output error be Q(t)=0.5(t*p-y)^T G(t*p-y), where y is the missing contribution and p its prediction. The initial slope is -y^T Gp. Euclidean reconstruction instead weights directions equally. Two constructed examples use y=(1,1), G=diag(1,100), and p=(1,2.2) or p=(1,-.2). Both reduce Euclidean squared error from2 to1.44 while increasing full-step Q from50.5 to72. But initial slopes are-221 and+19 respectively: overshoot versus immediate directional harm. Analytic slopes match finite differences. These are illustrative constructions, not fitted native models.

Thus a harmful full repair cannot by itself distinguish an initially useful direction with excessive magnitude from a direction that is immediately harmful. Overshoot can happen with linear readers and a quadratic metric; observing it does not establish downstream nonlinearity.

## Frozen native assay

At each MLP and each half mask, replace the write by retained_write + t*predicted_missing_write + bias. The prediction is fixed from the original retained write. Test t=-.01,+.01,-.001,+.001,+.1 on all160 task endpoints and32 natural evaluation documents. Reuse audited t=0 from M41 and t=1 from M42. No new task-holdout exposure. All downstream computation remains live.

Estimate central output-KL slopes at scales.01 and.001. Scale agreement requires their absolute difference divided by max(abs(slopes),1e-8) to be at most.1. Keep ambiguous cases. These are finite numerical estimates, not exact derivatives.

## Results

Output KL averages both masks and all positions/inputs within each cohort. Counts refer to input–mask cases, not independent documents or statistical confidence.

| Cohort | Layer | KL at deletion | KL at 10% repair | KL at full repair | Stable initially helpful / all cases |
|---|---|---:|---:|---:|---:|
| task train | MLP0 | 0.04038 | 0.03002 | 0.02568 | 263/264 |
| task train | MLP1 | 0.03689 | 0.03462 | 0.05247 | 230/264 |
| task train | MLP2 | 0.02303 | 0.02198 | 0.02380 | 228/264 |
| task evaluation | MLP0 | 0.03215 | 0.02474 | 0.02643 | 56/56 |
| task evaluation | MLP1 | 0.03738 | 0.03690 | 0.06308 | 39/56 |
| task evaluation | MLP2 | 0.02350 | 0.02312 | 0.02788 | 43/56 |
| natural evaluation | MLP0 | 0.02885 | 0.02246 | 0.02866 | 64/64 |
| natural evaluation | MLP1 | 0.03282 | 0.03121 | 0.06474 | 61/64 |
| natural evaluation | MLP2 | 0.02264 | 0.02212 | 0.03338 | 60/64 |

Small-scale estimates agree in1,148/1,152cases;1,044 have a stable helpful slope and104 a stable harmful slope. Four remain scale-ambiguous. Among682cases harmed by full repair,578 initially improve with scale agreement. Local Euclidean error has a helpful small-step slope in all1,152cases, so the104 stable harmful output slopes directly exhibit a local-metric/output mismatch.

MLP1 natural text is especially clear:61/64cases initially improve, all64 worsen at full strength, and pooled KL drops from.03282 to.03121 at10% before rising to.06474 at full strength. Every one of the nine layer/cohort aggregates improves at the fixed10% point. This is a diagnostic comparison, not a selected optimal amplitude or proof of a reusable repair policy.

## Conceptual update and stopping rule

The evidence now supports a graded account: halves share predictable information; identity doubling usually fails beyond MLP0; learned predictions tend to supply initially useful directions; their reconstruction-trained magnitude often fails downstream. The dominant observed failure is compatible with overshoot, while a smaller set has immediate directional mismatch. Neither observation establishes identical semantic subcomputations, general cancellation, or successful full functional replacement.

This closes the bounded half-repair diagnosis. Do not search for a best amplitude or refit these maps on evaluation outputs. The concrete featurizer lesson is to evaluate causal usefulness and calibration separately from local reconstruction and shared support. Any next fitting must be justified by a fresh training-only mechanism test and preserve task holdouts. A broader hypothesis synthesis should determine the next experiment rather than continuing scalar calibration indefinitely.

## Audit

5,952native forwards/5,760new intervention rows. All inventories, summaries, native source energies and task outputs replayed;30sampled NumPy predictions exactly match, output metrics differ by at most1.19e-8. The analysis joins audited M41/M42 anchors by exact input/layer/mask keys and source hashes. Numerical two-scale agreement does not replace independent transformer validation or cover arbitrary data distributions.

[Protocol](validation/native_half_interpolation_protocol.json) · [Toy examples](validation/native_half_interpolation_toy.json) · [Native results](validation/native_half_interpolation.json) · [Audit](validation/native_half_interpolation_audit.json) · [Slopes and case counts](validation/native_half_interpolation_analysis.json)
