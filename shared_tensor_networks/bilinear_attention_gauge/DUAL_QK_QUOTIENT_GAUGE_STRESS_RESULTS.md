# Environment-aware QK reduction survives exact gauges of trained heads

The [bond-gauge stress control](THREE_COUNT_BOND_GAUGE_STRESS_RESULTS.md)
showed that local factor energy can hide a task-critical direction while the
complete function stays fixed. Here we test the same issue on the **QK1
branch of ten trained, necessary-two-head bilinear-attention language
models**. Both heads are required for their two-bit retrieval task; all ten
full models solve the rare held-out combination. This extends the earlier
small-checkpoint WTG gauge check to independently trained task models,
output-path errors, a branch intervention, and held-out task behavior.

For each physical head, we extract its exact QK1 token score and factor it
into a rank-two quotient `U Vᵀ` on the full role-valid domain of two query
tokens and six key tokens. The learned physical QK bond is width four; the
rank-two quotient is an exact *domain restriction*, not a gauge transform of
the original width-four weights. We then apply six invertible `2×2` changes
of quotient coordinates per head: the identity, two one-axis suppressions,
and three randomly rotated suppressions. Each transformed `U,V` pair has
the same QK1 score and, with QK2 and OV untouched, the same complete native
head function to less than `1e-10` absolute error.

At bond rank one, compare two answer-free procedures under a uniform
role-valid query/key measure:

* **Local key SVD:** keep the largest right-singular direction of the
  transformed key factor `V`. To make this a stronger control, refit its
  scalar amplitude using the full output-weighted environment.
* **Environment WTG:** construct the exact QK1-bond environment including
  QK2 and OV, then truncate in the weighted-trace gauge. No task labels or
  rare-combination outcomes enter either fit.

| Rank-one method and quotient gauge | Mean head-output pair MSE | Mean QK2→constant pair MSE | Two-head rare held-out accuracy |
|---|---:|---:|---:|
| Local key SVD, original | .0146 | .0149 | .90 |
| Local key SVD, suppress first axis | .0146 | .0149 | .90 |
| Local key SVD, suppress second axis | **.4524** | **.5150** | **.65** |
| Local key SVD, three rotated gauges (mean) | .0714 | .0743 | .90 |
| Environment WTG, **every gauge** | **.0103** | **.0106** | **.90** |

The local method's within-head ordinary-error range across six exact gauges
has median `.304` and maximum `1.000` over the 20 trained heads. Its
rank-one token function can change by up to **14.58** in absolute output
units under a gauge that leaves the uncut function unchanged. WTG's maximum
corresponding output difference is `1.2e-12`, and its within-head error
range is below `4e-16`. Thus the WTG result is stable as a **function**,
not merely similar in an aggregate score. The QK2→constant intervention
was excluded from fitting; WTG happens to transfer well to this edit, but
that does not imply transfer to other branch interventions.

The WTG rank-one task result is 9/10 rather than 10/10 because one trained
seed loses the rare rule at this tight per-head rank. Exact gauge handling
does not repair information discarded by the reduction metric and budget.
The experiment also does not provide a canonical semantic decomposition
across heads: WTG fixes a *specified bond in a specified topology*. The
next step toward legibility is to compare sparse, response-based, and
intervention-based coordinates **within a fixed, faithful reduced
function**, then test selective finite effects on new task families.

Reproduce with `dual_trained_qk_quotient_gauge_stress.py`. Per-seed head
errors, gauge checks, branch edits, and exhaustive task outcomes are in
`dual_trained_qk_quotient_gauge_stress.json`.
