

## H421 — Unmatched mean/covariance controls

Reuse H420 recipients and use the disjoint H413 panel as a fixed donor calibration pool (512 documents each, full-document hashes independently verified). No new documents or group selection. Head6 and H410 core64/added64/union128 remain fixed. At each of four token positions, compute donor mean and population covariance in the native 128-dimensional head output subspace. Compare six raw donor bijections with six Gaussian draws from those moments. Apply no norm, angle, or other post-sampling matching. The sampled Gaussian batches match moments in expectation, not exactly.

The frozen three contrasts are Gaussian-minus-raw local cross/full energy, finite combination score S=R_union−R_core−R_added, and finite union recovery. A joint max-absolute 95% band uses 4000 paired recipient-document/draw bootstrap samples. Each equivalence gate requires its entire joint interval to lie within ±0.05. These intervals condition on the fixed donor calibration pool and do not include uncertainty in estimating population moments. Reused recipients and previously selected groups make this a mechanistic follow-up, not fresh confirmation. Raw donor permutations preserve donor dependence across positions; Gaussian samples are independent across positions, so marginal expected scores are comparable but document-level variability can differ.

|Condition|Local cross/full energy|Finite combination score|Union recovery|
|---|---:|---:|---:|
|actual|-0.228676|0.278367|0.803000|
|raw_donors|-0.248094|0.237521|0.814691|
|unmatched_gaussian|-0.260861|0.229538|0.816614|

|Gaussian-minus-raw contrast|Estimate|Joint 95% interval|Equivalence ±0.05|
|---|---:|---|---|
|Gaussian-minus-raw local cross ratio|-0.012767|[-0.040885, 0.015351]|True|
|Gaussian-minus-raw finite combination score|-0.007983|[-0.036101, 0.020135]|True|
|Gaussian-minus-raw union recovery|0.001923|[-0.026195, 0.030041]|True|

At a fixed recipient, write the source-to-group tangent response as A_g h and the native output Fisher metric as F. For donor mean μ and covariance Σ, E[2 hᵀ A_coreᵀ F A_added h]=2 tr(A_coreᵀ F A_added Σ)+2 μᵀ A_coreᵀ F A_added μ. The same identity applies to full energy. Therefore ratios of pooled expected quadratic quantities agree for moment-matched source distributions. This is an architectural/statistical identity, not evidence of semantic features. Finite Monte Carlo ratios need not agree exactly. Finite RMS/bilinear/output-softmax predictions can depend on higher moments and are the informative extension.

Outcome: all three frozen equivalence gates pass. Gaussian-minus-raw local cross ratio is −0.012767, joint interval [−0.040885, 0.015351]; finite combination-score difference is −0.007983, [−0.036101, 0.020135]; union recovery difference is +0.001923, [−0.026195, 0.030041]. Union recovery is 0.814691 for raw donors and 0.816614 for Gaussian samples. Both retain a negative local cross ratio and positive finite combination score. The actual-source condition exactly replays H420 arrays.

This changes the interpretation of H414–420: failure of moment controls after norm/angle matching does not imply that richer natural source structure is required. Without that transformation, first and second moments suffice to reproduce the three pooled scores to the declared resolution on this panel. H420's component recombination remains a valid intervention, but its finite effect can plausibly arise from altered mean/covariance structure rather than a special semantic association. This experiment does not directly fit or compare the moments of H420's recipient-dependent recombined distribution, so that specific mediation claim remains unproven.

Update to the twenty-hypothesis tournament: #12 (operator/output alignment) strengthens; #5 (compensating combinations matter) remains true but need not imply a discrete semantic pair; #17 gains a finite, aggregate second-moment explanation at unit amplitude; #19 (semantic features) receives no support. #20 (context-conditioned response operators) remains promising. Higher-order structure is not ruled out for individual examples, other sources, larger interventions, or linguistic tasks. Equivalence is limited to three aggregate metrics, fixed selected groups, the donor pool, and the stated tolerance; these are not interchangeable source distributions.

Next discriminating experiment: an amplitude sweep using the same frozen natural and Gaussian source samples. At small amplitude the local moment identity should predict agreement; larger amplitudes may reveal higher-moment dependence through input RMS, bilinear terms and output geometry. Predeclare amplitudes and a simultaneous contrast family, include both absolute effect size and normalized recovery, and preserve a separate fresh confirmation if a meaningful discrepancy emerges. Another useful branch is an exact conditional second-moment prediction for H420 recombination: its transformation depends on the recipient, so global covariance alone may be insufficient even though recipient-conditional moments determine each quadratic statistic. These tests distinguish nonlinear distributional dependence from conditional operator geometry before attaching semantic labels.


Audits: {'clamp_state_error': 5.841255187988281e-06, 'covariance_root_error': 3.259629011154175e-09, 'donor_projection_error': 3.788849143318059e-07}. Independent saved-array checks: {'actual_replay_max_abs': {'tangent': 0.0, 'finite': 0.0, 'effect': 0.0}, 'covariance_factor_max_abs_error': 3.259629011154175e-09, 'covariance_factor_relative_error': 1.3621185193420218e-15, 'all_maps_bijective': True, 'pooled_scores_and_joint_band_replayed': True}.

The broad research goal remains open; completed unique documents remain 9408. No semantic labels or universal computational units have been established.
