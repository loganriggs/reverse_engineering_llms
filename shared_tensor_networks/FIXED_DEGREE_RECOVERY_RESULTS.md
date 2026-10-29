# Factor recovery, canonical interventions, and repeated-factor instability

Assessed all81previous fixed-degree fits without retraining. Each teacher/student is expanded into four linear-factor occurrences, retaining multiplicity. Enumerate all24permutations to maximize mean absolute cosine; report worst matched cosine, independently optimal bottleneck cosine, sign-corrected global amplitude, and initialization baseline. Hungarian assignment independently agrees with the exhaustive mean optimum. Permutation, reciprocal scaling, sign reversal and wrong-multiplicity fixtures pass.

Descriptive strict recovery criterion, fixed before this assessment: every matched occurrence cosine>=.99 and signed global amplitude within5% of teacher. It is not a significance test. Zero-norm factors are invalid, rather than silently normalized. Directions are invariant under individual nonzero scaling/sign; original global amplitude is retained to detect sign or magnitude errors.

## Generic four-factor target, four-factor student, independent initialization

| Seed | Exact population MSE | Worst factor angle | Signed amplitude ratio | Recovery criterion |
|---|---:|---:|---:|---|
| 170001 | 0.00130433 | 2.304 degrees | 0.98257 | pass |
| 170002 | 0.000453046 | 2.837 degrees | 1.02117 | pass |
| 170003 | 9.18493e-05 | 0.765 degrees | 1.00107 | pass |

All3pass. Initial mean factor cosines were.795,.664,.703; the good final match was not already present at initialization. Minimum distinct teacher-factor angles are31.70,15.63,35.87degrees. This supports approximate factor recovery in the specified product grammar, not recovery of an arbitrary physical execution DAG.

## Recovery is not uniform across families

For independently initialized correctly specified student families, strict recovery passes3/3 for one-factor teachers,0/3 for two-factor teachers,3/3 for four-factor teachers. The two-factor models retain finite optimization error. Overparameterized four-factor students on one-factor teachers pass0/3 with independent initialization, despite a median prediction MSE around.0053. Small perturbations pass2/3 on that family.

The earlier one-factor symmetric-initialization median nearzero concealed one failure: seed170001 has MSE.940894, while the other two are around1e-13. That seed fails identically for all symmetric studentgroup counts, as predicted by the invariant-trajectory theorem. All cases remain in the records; do not describe median success as universal recovery.

## Canonical factor interventions and covariance shift

For all3independently initialized four-factor/four-factor cases, normalize each factor to unit length, oracle-align its sign and permutation to the teacher, and retain original overall amplitude. Delete one canonical factor by replacing it with1. This gives four cubic intervention outputs per model. Teacher alignment is privileged evaluation, never training supervision; these are interventions on a specified mathematical representation, not a claim about a hidden physical graph.

Across12factor deletions, final relative Gaussian MSE ranges0.0000641–0.004564, compared with0.344–2.715 at initialization. Under covariance diag(.5,1,2), final relative function MSEs are.003118,.0001097,.00003979. Independent5-point-per-axis quadrature replays all36initial/final intervention and shifted-function moment calculations with maximum relative discrepancy1.257e-14. This is conditional evidence beyond ordinary prediction accuracy, but only for these3targets and this canonical intervention definition.

## Why repeated factors can be harder to recover than functions

Take independent standard Gaussian u,v and common normalization sqrt105. Define

f0=u^4/sqrt105,
f_delta=(u+delta*v)(u-delta*v)u^2/sqrt105.

Then f_delta-f0=-delta^2*u^2*v^2/sqrt105, and exactly

E[(f_delta-f0)^2]=(3/35)*delta^4.

The two moved factors differ in direction by arctan(|delta|), which is first order in delta. Thus factor displacement is order sqrt(function L2 error), or fourth-root of squared prediction error. There is no uniform Lipschitz inverse from function L2 distance to factor directions at this repeated-factor collision. Exact identifiability by unique factorization does not imply stable approximate recovery.

The common normalization is deliberate; this fixture does not independently renormalize f_delta. Coefficient moments and angle formulas were checked at five perturbation magnitudes. This is a classical root-conditioning phenomenon illustrated in the current model family, not a new general identifiability theorem. Near-collision conditioning is a candidate explanation, not a proved exclusive cause of every observed recovery failure.

## Next

This closes the immediate question of whether the successful generic fits recover factors: they do, approximately, with canonical intervention agreement. Preserve the failures and conditioning distinction. Next compare sample-size and observation-noise effects with exact population/factor evaluation retained, and vary factor separation explicitly if attributing recovery differences to conditioning. Do not keep extending one favorable target family as a substitute for broader scaling or general circuit recovery.
