## H437 — Small target-free omitted operators

H436 established that dense analytic local prediction recovers the omitted directions. H437 restricts that local operator to the native128-dimensional head6 source subspace and approximates it at ranks8,16,32,64. Directions are chosen from baseline states, model weights and frozen donor second moments; finite target responses are not used to select directions or fit coefficients. Evaluate exact union128 finite response plus the approximated omitted correction on the same H434 panel. No new documents; total11968.

The source second moment is covariance+mean*mean^T from H421, including nonzero source mean. Its square root defines whitened source coordinates. Compare local output-Fisher eigendirections, local raw-output eigendirections, shared leave-one-context-out Fisher directions, and one fixed random orthobasis. Shared bases average trace-normalized source-space Grams from the other31 contexts. They are shared input/source directions with context-dependent output writes, not a universal output basis. All methods predict coefficients from the actual source input rather than fitting the full response.

| Frozen rank16 contrast | Estimate | Joint95% interval | Gate passes |
|---|---:|---|---|
|rank16_local_low|0.967062|[0.874589, 1.059535]|False|
|rank16_local_median|0.988816|[0.896343, 1.081289]|False|
|rank16_local_minus_shared_low|0.045215|[-0.047258, 0.137687]|False|
|rank16_local_minus_shared_median|0.017935|[-0.074537, 0.110408]|False|

All four joint gates fail: >90% omitted-Fisher coverage in each stratum and >5-point local-over-shared advantage in each stratum. The point estimates are promising, but these intervals do not establish the predeclared guarantees or a local advantage. Intervals use4000 paired source/document resamples on reused selected strata, conditional on the donor pool; they are not new independent confirmation. Symmetric bands can extend above the physical recovery ceiling1.

| Method | Rank | Low-RMS full finite-KL recovery | Median-RMS full finite-KL recovery |
|---|---:|---:|---:|
|local_Fisher|8|98.450%|99.760%|
|local_Fisher|16|98.550%|99.827%|
|local_Fisher|32|98.611%|99.880%|
|local_Fisher|64|98.655%|99.916%|
|local_raw|8|97.128%|99.506%|
|local_raw|16|97.558%|99.617%|
|local_raw|32|97.824%|99.727%|
|local_raw|64|98.236%|99.863%|
|shared_leave_one_out_Fisher|8|96.788%|99.465%|
|shared_leave_one_out_Fisher|16|97.011%|99.545%|
|shared_leave_one_out_Fisher|32|97.830%|99.674%|
|shared_leave_one_out_Fisher|64|98.294%|99.854%|
|random|8|59.524%|85.195%|
|random|16|57.358%|84.706%|
|random|32|46.448%|80.276%|
|random|64|54.841%|84.293%|

At rank8, local-Fisher full recovery reaches98.45% low/99.76% median versus61.10%/84.67% for union128 alone. Shared leave-one-out directions reach96.79%/99.46%. Dense linear correction previously reached98.67%/99.93%. These descriptive values show that much of the dense predictor’s useful action fits in a few input-derived directions, but do not establish the superiority of local directions or generalization to independently selected contexts/sources. Random source-coordinate directions are substantially worse in these finite samples, and increasing their rank is not monotone under finite KL. No failed gate is relabeled.

Full128-direction operator prediction replays the prior dense analytic omitted predictor within6.19e-07 relative vector discrepancy, below1e-5. The residual discrepancy is consistent with native-float head QR span precision. Positive source-second-moment eigenvalues permit inversion without a tuned cutoff. Scored arrays are finite.

**Representation versus computation:** after construction, a rank-r operator can store128*r source-read coefficients and1152*r output-write coefficients per context, plus shared coordinate transforms. Rank8 needs10,240 coefficients versus147,456 for a dense1152x128 local operator (6.94%); rank16 needs20,480 (13.89%). This excludes the retained128-factor computation, baseline model, shared QR/second-moment transforms and basis-construction work. Building the operator still reads all4608 factors and uses4480 omitted output factors. Local-Fisher construction also applies the output metric and performs an eigendecomposition. This is compact conditional representation, not demonstrated inference acceleration or global weight compression.

The result narrows the question: small input-output maps can predict consequential omitted directions without fitting finite targets. It does not yet show that those maps are stable reusable semantic units, cheap to construct, or transferable as one frozen representation. Shared-direction performance is a live alternative to the claim that every context needs a distinct basis.

Next freeze one shared source-direction basis from this calibration panel, then evaluate independently selected contexts and fresh source draws without recomputing the shared basis. Compare rank8/rank16 shared predictions, local rank-matched predictions and dense analytic ceilings. Keep finite KL, omitted-error coverage and total construction/storage costs separate. If shared directions transfer, investigate a compiled way to produce their context-dependent output writes; if they fail, test which baseline geometry predicts the failure rather than selecting from target errors.
