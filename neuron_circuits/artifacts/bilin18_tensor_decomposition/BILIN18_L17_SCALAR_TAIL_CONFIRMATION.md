## H450 — Scalar tail strength and mean-contrast confirmation

Use128 receiving contexts whose intervention outcomes were unused, excluding288 previously tested cases from the H438512-document panel. Use24 fresh source draws and eight fresh random spans; learned span, native union128 and radial correction stay frozen. Baselines had been collected, so this is independent receiving-intervention transfer within an existing corpus panel, not new corpus documents. Total12480;96 intervention-unused contexts remain in this panel. Sources remain conditional on the fixed donor pool.

Compute the exact conditional tail covariance trace at each of128 disjoint H444 calibration baseline states. Their unweighted average is the frozen constant. Cross receiving-specific versus constant isotropic tail strength with exact versus omitted head/tail between-group covariance and cross-covariance. In every condition retain exact top128 constraints, receiving tail mass, output-RMS transport and condition cap1000; include full and top-only references. The constant/no-between variant does not require tail gradient means or receiving-specific tail covariance trace for fitting, although the runner computes them for controls.

Calibration mean trace=6.72226388. Receiving trace min/median/max=3.300912/6.792982/8.686888. Trace includes the1/2 Fisher convention and is measured before receiving RMS transport.

| Condition | Learned finite KL recovery | Random mean recovery | Learned included / omitted Fisher error | Random included / omitted Fisher error |
|---|---:|---:|---|---|
|full_condition|98.166%|95.744%|0.010190 / 0.006958|0.018388 / 0.021589|
|top128_condition|97.562%|94.541%|0.010313 / 0.013123|0.015545 / 0.036135|
|isotropic_own_between|98.134%|95.613%|0.010230 / 0.007000|0.018483 / 0.022679|
|isotropic_constant_between|98.131%|95.599%|0.010168 / 0.007093|0.018356 / 0.022974|
|isotropic_own_no_between|98.079%|95.230%|0.010357 / 0.007363|0.018411 / 0.026100|
|isotropic_constant_no_between|98.074%|95.237%|0.010296 / 0.007463|0.018237 / 0.026263|

| Frozen contrast | Estimate | Joint95% interval | Lower > −.005 |
|---|---:|---|---|
|isotropic_minus_full_learned|-0.000312|[-0.002417, +0.001793]|True|
|isotropic_minus_full_random_mean|-0.001318|[-0.003423, +0.000787]|True|
|constant_no_between_minus_own_between_learned|-0.000601|[-0.002706, +0.001504]|True|
|constant_no_between_minus_own_between_random_mean|-0.003758|[-0.005863, -0.001653]|False|

Four prospective noninferiority gates use a0.5pp margin: full-context isotropic versus full exact, and constant/no-between versus own-trace/exact-between, each learned/randommean. Joint95% bands use4000 paired source/context resamples and eight-span resampling. Calibration uncertainty is not resampled. Separate intermediate factorial effects are descriptive.

Audit: relative solve residual4.15e-15, exact full covariance decomposition error2.89e-15, largest regularized condition number1000.000000; all gates pass. Arrays finite, input hashes checked and selection disjointness verified. No exclusions.

The isotropic within-tail correction confirms on unused receiving interventions: own-trace/exact-between recovery98.134% learned and95.613% randommean, versus98.166%/95.744% full. Both frozen0.5pp noninferiority gates pass; deficits are0.0312pp and0.1318pp with joint95% intervals[−0.2417,+0.1793] and[−0.3423,+0.0787]pp. This establishes the tested practical tolerance, not equality or a whole-model claim.

Replacing the receiving trace by one calibration constant while keeping the between-group term changes point recovery by only−0.0037pp learned and−0.0135pp random. No standalone constant-trace equivalence gate was frozen, so this small difference remains descriptive. Constant trace/no-between recovery is98.074% learned and95.237% random. Versus own-trace/exact-between, the learned simplification passes noninferiority at−0.0601pp (joint95%[−0.2706,+0.1504]); the random simplification fails at−0.3758pp ([−0.5863,−0.1653]). The random interval indicates a loss, while the practical0.5pp boundary remains unresolved.

The factorial points locate most simplification cost in omitting the between-group mean contrast, not replacing tail strength: own-trace/no-between is98.079%/95.230%, nearly the same as constant/no-between. This attribution is descriptive, not a separately confirmed interaction test. All covariance variants retain exact likely-token constraints and RMS transport. A scalar isotropic tail is supported at the tested tolerance; completely removing tail mean information is not yet supported across both span classes.

Updated explanation: generic within-tail constraints plus a compact between-group mean correction can replace a large conditional tail covariance for these interventions. The surviving rank-one mean term concerns the difference between weighted head/tail average logit gradients, not necessarily a human semantic feature. Next measure and approximate that tail mean from disjoint baseline calibration: compare exact receiving tail mean, shared mean, and zero mean while preserving the head mean, constant isotropic tail, mass and condition rule. This will distinguish genuinely context-specific tail direction from a reusable offset. Freeze the approximations before evaluation and count the retained dense analytic source map separately. The broader question still requires transfer across sources/layers and an account of interpretable computational content, not just output approximation.

Scope/cost: one calibration scalar replaces storage of a full tail covariance in the constant isotropic variant. Exact top-token probabilities, tail mass, receiving RMS geometry and dense analytic response still enter. Neither a speedup nor an independent semantic feature decomposition has been established.
