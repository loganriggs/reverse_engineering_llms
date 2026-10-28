

## H420 — Fresh component recombination changes finite group roles

Fresh 512 documents exclude all 8896 earlier documents, bringing the completed total to 9408. Freeze H410 core64, added64 and union128, head6, four token positions and six donor-pair maps. Raw donor A is compared with a same-donor reconstruction and a crossed reconstruction preserving A’s norm and recipient-relative angle while taking its orthogonal direction from donor B. A and B mappings are bijections, exclude self donors, and differ at every recipient. The transformation preserves those scalar constraints; it does not preserve the source distribution or covariance.

Use double local derivatives with native RMS epsilon and native finite edits. The finite score is S=R_union−R_core−R_added, with each recovery R=1−pooled KL(native edited || group prediction)/pooled KL(native edited || intact). This is a relationship among prediction scores, not a unique causal mediation fraction or a nonlinear forward interaction. A joint max-absolute 95% band covers three preregistered contrasts using 4000 paired document/mapping bootstrap draws, conditional on this panel and mapping design.

|Condition|Local cross/full energy|Core recovery|Added recovery|Union recovery|Combination score S|
|---|---:|---:|---:|---:|---:|
|Actual|-0.228676|0.647521|-0.122889|0.803000|0.278367|
|Raw donor|-0.254660|0.677029|-0.109528|0.810569|0.243068|
|Self reconstruction|-0.254660|0.677029|-0.109528|0.810569|0.243068|
|Recombined|0.052307|0.798686|0.125203|0.872074|-0.051815|

|Frozen contrast|Estimate|Joint 95% interval|
|---|---:|---|
|cross_minus_raw_full_cross_ratio|0.306967|[0.194801, 0.419134]|
|mixed_numerator_shift_over_raw_full_energy|0.720367|[0.608201, 0.832534]|
|raw_minus_cross_finite_combination_benefit|0.294882|[0.182716, 0.407049]|

All three scientific lower bounds exceed the frozen 0.10 threshold. Document disjointness, geometry and self-reconstruction gates also pass. Audits: {'norm_error': 3.827210364508708e-07, 'dot_error': 3.2590761069389484e-07, 'span_error': 5.020757917629364e-07, 'self_vector_error': 5.541720256740321e-07, 'clamp_state_error': 6.4373016357421875e-06, 'self_pooled_score_error': 2.424659917288352e-08}. Saved-array checks independently reproduce mapping constraints, pooled scores and the joint band.

Interpretation: joint directional structure matters for this selected source and these factor groups, including finite predictions. Raw shuffling preserves negative core/add cross and positive combination score; breaking the association between scalar geometry and orthogonal direction reverses both. Same-donor reconstruction matches raw scores within 2.43e-8. This supports a functional role for the association rather than a reconstruction implementation artifact.

The union recovery rises from 0.811 to 0.872, while the combination score falls from 0.243 to −0.052. Thus “loss of combination benefit” must not be described as loss of overall prediction quality. Recombined sources also have larger full response energy. The mixed numerator contrast uses the common raw-energy denominator and remains positive, so differing denominators alone do not explain that contrast. Recombination changes the joint source distribution and can change second moments; higher-order semantic organization is not established. Native source-recipient pairing was already unnecessary for the raw-shuffle effect.

Priority update to the H419 twenty-hypothesis tournament: strengthen #5 (combinations matter), #12 (output/operator alignment), and #17 (this local relationship has a finite counterpart); weaken #10 (single-panel dependence) for this selected mechanism. #7 (norm/angle alone) remains insufficient. #19 (semantic features) remains unsupported. #20 (context-conditioned response operators) remains a useful descriptive candidate, not an identified universal unit. All other hypotheses retain their prior status.

Next high-information test: compare raw donor samples with Gaussian samples having their untransformed empirical mean and covariance, without any norm/angle matching. At a fixed recipient the derivative is linear in the source, so its expected quadratic Fisher statistics depend only on first and second moments. Check this local prediction explicitly, then ask whether finite predictions depart from it. This separates a second-moment/operator explanation from higher-order distributional structure; failure after additional geometry matching cannot answer that question. Keep calibration and evaluation separate and quantify Monte Carlo uncertainty. A second useful test varies intervention amplitude to distinguish the quadratic approximation from finite nonlinear departures.

The broad research goal remains open. Future fresh panels must exclude 9408 completed-panel documents.
