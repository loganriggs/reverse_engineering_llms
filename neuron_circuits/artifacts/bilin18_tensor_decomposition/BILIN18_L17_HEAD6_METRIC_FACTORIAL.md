

## H349 — RMS normalization and token-probability weighting

Reuse H346 recipients and cached exact finite compiler coefficient deltas. Cross two metric factors: full RMS derivative versus scaling-only derivative (omit radial subtraction), and native token-probability covariance versus uniform-vocabulary covariance. Keep the capped-logit derivative in all four cells. The scale-only and uniform conditions are metric diagnostics, not alternative model runs.

At each site and realization divide metric response energy by Euclidean residual-output response energy, then average equally across sites and realizations. Report control/actual ratios of these output Rayleigh averages. This statistic differs from H346 normalized input gain and H348 aggregate KL ratios; their numerical values should not be equated.

|Metric|Moment control/actual [95% interval]|Donor control/actual [95% interval]|
|---|---:|---:|
|radial_prob|1.846 [1.515, 2.227]|1.768 [1.455, 2.154]|
|radial_uniform|1.442 [1.335, 1.553]|1.376 [1.280, 1.481]|
|scale_prob|1.274 [1.038, 1.521]|1.230 [1.015, 1.468]|
|scale_uniform|1.118 [1.058, 1.182]|1.081 [1.033, 1.133]|

Under full RMS, replacing uniform covariance with native probability covariance increases the control/actual ratio by moment: 1.280 [1.072, 1.512]; donor: 1.285 [1.080, 1.524]; both paired intervals exceed1 but fall far short of a2× effect. Full RMS radial subtraction also increases the ratio relative to scaling alone. Neither the radial diagnostic nor the probability diagnostic is a causal percentage of the observed effect; their interaction remains possible.

Locked gates: probability_amplification2: FAIL, uniform_gap_under1_5: FAIL. Probability amplification lower95%>2 fails for both controls. Uniform-gap upper95%<1.5 fails jointly because the moment upper bound is1.553 (donor1.481). These negative results constrain a probability-weighting-only explanation.

Uncertainty uses4000 shared document resamples and global realization resamples with paired comparisons across metric cells. Intervals are pointwise and conditional on the fixed compiler, donor assignments and six realizations; donor dependence is not reconstructed. Same panel as H346, with no new replication or semantic claim.

Audit: full-RMS native-probability metric replays H346 with relative error 3.39e-16. Cached source hashes checked. Per-site Euclidean energies, four metric energies and metric matrices saved. Native weights unchanged.

Next: decompose the RMS radial subtraction into scaling, radial correction and their signed interaction, using exact output derivatives. Determine whether natural compiler responses are preferentially radial or whether cancellation with the radial correction matters. Then revisit a broad hypothesis set rather than promoting one metric-dependent direction into a semantic feature. Broad research goal remains active.
