## H463 — Does covariance explain natural source compactness?

Head6 only. Exclude previous H459 donor IDs, draw64 evaluation donors without replacement. Recalibrate mean, centered source basis16 and covariance on remaining donors excluding both sets. Freeze before outcomes. Natural centered, centered covariance-Gaussian and isotropic latent native-weight controls, synthetic outputs norm-matched per natural donor.128 unused H454 recipient sites excluding H454/H458 selections. Primary covariance-minus-natural data16 equivalence within5pp each sign, and covariance-minus-isotropic improvement above20pp each sign; joint4 bootstrap bands. Fixed calibration uncertainty. No semantic interpretation from covariance sufficiency.

The centered rank16 basis and Gaussian covariance are newly calibrated on the donor split excluding64 evaluation donors and the prior H459 donor IDs. This prevents calibration leakage for the new evaluation; it is not an exact replay of the H462 basis. Evaluation donors have earlier baseline/calibration exposure, and the corpus is reused. Receiving sites are intervention-unused within the H454 panel. Total tracked corpus documents remains12992.

The Gaussian control samples the empirical centered calibration covariance in expectation before per-sample output-norm matching. Norm matching alters covariance; empirical before/after covariance audits are saved. Neither Gaussian control is asserted to be naturally reachable. The same64 natural-source norms are used across all three families.

| Family | Negative data16 / retained-only recovery | Positive data16 / retained-only recovery | Source energy lost by data16 |
|---|---|---|---:|
|natural_centered|99.386% / 85.980%|99.327% / 84.570%|1.561%|
|covariance_gaussian_normmatched|94.651% / 79.024%|95.335% / 80.914%|12.454%|
|isotropic_latent_normmatched|61.696% / 50.217%|55.937% / 43.996%|81.232%|

| Contrast | Estimate | Joint95% interval | Pass |
|---|---:|---|---|
|covariance_minus_natural_negative|-0.047358|[-0.142460, +0.047744]|False|
|covariance_minus_natural_positive|-0.039923|[-0.135024, +0.055179]|False|
|covariance_minus_isotropic_negative|+0.329545|[+0.234443, +0.424646]|True|
|covariance_minus_isotropic_positive|+0.393979|[+0.298877, +0.489080]|True|

Equivalence requires the entire joint interval inside[-.05,+.05] for covariance-minus-natural at each sign. Superiority requires the lower bound>.20 for covariance-minus-isotropic at each sign.4000 paired source-index and recipient-context bootstrap replicates; covariance calibration and selected source basis remain fixed. No thresholds were changed after outcomes.

Covariance-shaped controls substantially improve on isotropic controls: negative/positive data16 KL recovery is94.651%/95.335% versus61.696%/55.937%, while natural centered sources reach99.386%/99.327%. Covariance-minus-isotropic gains32.954/39.398pp have joint95% intervals[23.444,42.465]/[29.888,48.908]pp, passing both frozen20pp superiority gates. Covariance-minus-natural gaps−4.736/−3.992pp have intervals[−14.246,+4.774]/[−13.502,+5.518]pp; neither passes the prespecified±5pp equivalence gate. These intervals do not prove a remaining natural-source advantage. The result supports a large role for second-order source geometry without establishing complete sufficiency.

The covariance audit identifies a concrete confound in interpreting any remaining gap. Before per-sample normalization, Gaussian controls lose only1.297% source energy outside data16. Independently pairing Gaussian directions with natural source norms raises this loss to12.454%, versus1.561% for natural signals. Matching covariance in expectation before normalization does not preserve covariance or direction/magnitude coupling after normalization. The empirical centered covariance Frobenius error relative to calibration changes from10.86% to16.61%; this aggregate measure also understates the large change in the small complementary component. Negative/positive Gaussian-to-natural total effect KL ratios are approximately1.014/1.166. Effective donor counts based on effect concentration are approximately15–19 for these families, despite64 donors, and the bootstrap keeps calibration fixed.

A post-hoc source-only diagnostic compares normalization policies on the same draws: sorting Gaussian vectors by original norm and assigning the sorted natural norms yields1.443% rank16 source error, while one global scale to match total natural energy leaves1.297%. Both preserve total source energy, as does the original paired policy. Neither alternative has been evaluated behaviorally yet. These diagnostic point estimates cannot be described as recovered intervention effects or as fresh confirmation.

The next discriminator is therefore normalization-policy dependence, before seeking semantic explanations: hold the Gaussian directions, receiving contexts, source basis and native retained factors fixed; compare independently paired norm matching, sorted norm matching and a single global scale at both signs. Freeze the prediction that sorted/global policies close the apparent gap and outperform paired matching; assess source energies, absolute probability effects and pooled recovery separately. This is a mechanistic test on a reused panel, followed by independent draws if successful. Sorted matching preserves the empirical norm distribution but changes its coupling to direction; global scaling preserves covariance shape up to scale but not the natural per-source norm distribution. Neither alone isolates semantics. No new labels, source basis or outcome-based fitting are required.

The exact retained128-factor response and native omitted MLP response to reconstructed sources remain part of these predictors. This tests distributional sufficiency of the source description, not a semantic feature ontology or a whole-model replacement. Numerical norm/disjointness audits and independent scalar recovery reconstruction pass.
