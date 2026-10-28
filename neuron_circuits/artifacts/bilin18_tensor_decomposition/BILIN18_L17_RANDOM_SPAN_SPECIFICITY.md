

## H377 — Random spans recover much of the behavior under response-dependent oracle fitting

Freeze three Gaussian QR random64 bases with seeds20270602/03/04, sign-corrected QR, and nested first16 columns. Reuse all512 H372 documents, three heads and13 actual/control responses. Each random span gets the identical H374 per-site Fisher coefficient fit,1e-12 relative eigencutoff, and H376 finite-KL grid safeguard. Compare against the saved probability-axis results. No best-seed selection; primary random statistic averages KL over all three orientations before normalization.

|Head|Family|Random16 recovery|Probability16 advantage|Random64 recovery|Probability64 advantage|
|---|---|---:|---:|---:|---:|
|0|actual|0.570|0.171|0.811|0.050|
|0|moment|0.508|0.056|0.725|0.022|
|0|donor|0.513|0.060|0.731|0.023|
|1|actual|0.565|0.088|0.767|0.032|
|1|moment|0.547|0.071|0.750|0.033|
|1|donor|0.549|0.071|0.753|0.033|
|6|actual|0.541|0.146|0.778|0.046|
|6|moment|0.573|0.081|0.821|0.025|
|6|donor|0.576|0.084|0.823|0.025|

Frozen gates: {'probability_advantage': False, 'probability_advantage10': False, 'random64_half': True}. Random64 exceeds50% recovery in all9 joint lower bounds; point recovery is72.5–82.3%. The universal probability-axis advantage gate fails because two rank64 intervals include zero. The universal10pp advantage gate also fails. All18 point advantages favor probability axes; all9 rank16 advantages have positive simultaneous lower bounds, but rank64 gains are only2.16–5.04pp and only7/9 lower bounds exceed zero. Preserve the failed universal criteria.

Rank16 random recovery is50.8–57.6%, versus56.4–74.1% for probability axes. Thus probability geometry has useful specificity at smaller rank, yet a large part of rank64 oracle recovery is also available in arbitrary spans. High oracle recovery alone is insufficient evidence that a subspace identifies model features or fundamental computational units. A plausible explanation is that context-specific fitting focuses on a limited set of likely-token effects; this mechanism has not been directly isolated by this experiment.

All methods use the complete native target response for coefficient fitting and finite step selection. No method here computes those coefficients from compact model inputs. This shared oracle access is essential to the comparison and prevents interpreting random-span recovery as a compressed native MLP. Guarding ensures no worse than baseline by candidate construction.

Audits: baseline KL replay is exact for all3 orientations; no local fit exceeds the corresponding Euclidean projection error; normal-equation residual maxima are below3.5e-13; all16/64 Gram directions retained. Finite KL and safeguard inequalities pass. Save each random basis, local/finite/grid errors, choices, coefficients, and paired bootstrap arrays. Frozen source SHA256 checks pass and native weights unchanged.

Uncertainty:4,000 paired recipient-document/control-draw resamples produce joint18 bands for mean-random recovery and separate joint18 bands for probability-minus-random gaps. These condition on the three fixed random bases, discovery geometry and empirical donor assignments. Three orientations provide a limited specificity control, not a calibrated random-subspace population distribution. Figure includes individual-seed point estimates, not confidence intervals. Reused panel; no fresh-document confirmation of the comparison.

Hypothesis update: probability axes carry some useful geometry, especially at rank16, but the claim that oracle success primarily demonstrates special feature directions is weakened. Generic moderate-dimensional subspaces plus context/response-dependent fitting explain much of the recovery. The coefficient-generation problem remains central.

Next discriminating analysis: measure random-span recovery versus intact distribution concentration using fixed entropy bins and document-level uncertainty. Then test whether replacing each recipient’s fitting metric with a shared or confidence-matched metric removes the generic recovery. This distinguishes fitting a small likely-token set from genuinely transferable computation, before pursuing semantic labels or coefficient predictors. Broader goal remains active.
