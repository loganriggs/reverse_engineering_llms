## H527 — Fresh directional confirmation supports KL, not yet the label contrast

Freeze two unit directions before new gradient collection: the pooled H512/H521 mean prediction-KL gradient and pooled mean observed-label contrast g_CE−g_KL. The rank36 read/write map is unchanged. Positive fresh projection onto a frozen gradient direction means its negative is locally descending for the corresponding objective/contrast. Contrast descent alone does not imply CE descent, because CE also contains the KL-gradient component.

Collect2,048 fresh documents and evaluate per-document gradients at unit gains. Prespecified primary statistics are the KL-gradient projection onto the frozen KL direction and contrast projection onto the frozen contrast direction. Use4,000 document bootstrap draws (seed20280109), fixed original paired standard errors and centered max-absolute standardized deviations for simultaneous95% two-sided intervals over these two statistics. No direction, step size, or block is selected on the fresh data.

| Statistic | Mean projection | Simultaneous lower | Simultaneous upper | Positive lower |
|---|---:|---:|---:|---|
|KL|0.00040934|0.00032225|0.00049642|True|
|CE_minus_KL|0.00098199|-0.00107920|0.00304318|False|

The frozen KL direction reproduces: mean projection0.0004093, simultaneous bounds[0.0003222,0.0004964]. The contrast estimate is positive0.0009820, but bounds[−0.0010792,0.0030432] include zero. This is a failed confirmation of that particular direction, not evidence of equivalence between CE and KL, no task-loss signal, or absence of a useful direction elsewhere in the36-dimensional gain family. A first-order gradient confirmation is also not proof that an arbitrary finite step improves held-out loss.

| Sequential block | KL mean projection | Contrast mean projection |
|---|---:|---:|
|1|0.00034426|-0.00174049|
|2|0.00045909|0.00055892|
|3|0.00046976|-0.00025970|
|4|0.00036423|0.00536924|

KL projections are positive in all four blocks. Contrast block means range−0.00174 to+0.00537 and change sign. These are random-sampling-order blocks, not identified semantic or temporal regimes. The final block must not be selected as a successful subgroup; all2,048 documents define the primary result. Block variation is compatible with sampling variation and does not itself establish heterogeneity.

Baseline rank36 replacement KL is0.00318677 and CE change+0.00382901 on this panel; these are descriptive means, not loss-neutrality tests. The experiment preserves the same rank budget, factors and gain parameterization. It does not identify unique computational axes or optimize CE.

Data handling: the previous cache had only1,789 unused eligible documents, so the2,048-document builder stopped before writing a partial panel. A new10,000×513 uint16 cache was built from FineWeb stream documents81,000–88,423, after the previous cache ended80,423. Shape, token range and SHA256 were independently validated. The cache tokenizer warning about long source text is unrelated to the model inputs, which use128-token windows. The fresh panel was sampled entirely from the new cache and checked against all24,064 prior tracked complete-document hashes, including H522 calibration. Independent rehashing verifies all windows and zero overlap. Total tracked unique documents26,112; this does not guarantee exclusion from the model's original training data.

Audits: map reconstruction replays exactly; head-write sum relative error4.34e−7; all per-document gradients and losses finite; frozen directions hashed before evaluation; independent bootstrap replay from saved gradients reproduces the simultaneous intervals. Gradient finite-difference and endpoint checks were established in H525. Both unit directions were formed from prior data only. No additional documents or outcomes were selected after inspecting this result.

Next H528: quantify a model-implied single-label noise reference on the H527 contexts without fitting new directions. At the compressed baseline input, compute directional logit derivatives for both frozen gain directions. Weight their centered values by the original model's full token probabilities to obtain exact conditional variances/covariance of the label contrast if a token were sampled from that original distribution. Compare observed target-token contrasts with this reference, replaying the H527 per-document projections. Report summed conditional variance/n² for the mean, observed contrast means, variance ratios, and concentration across contexts. Explicitly distinguish this teacher-distribution sampling model from the unknown true token distribution: agreement would support a noise-scale explanation, not prove model calibration or absence of task signal. The goal is to understand whether lack of CE reproducibility has the scale expected from one observed token per context, before spending more data or optimizing an unstable direction. No new documents or gain tuning.
