## H453 — Joint input distribution and map regression

Reuse H45296 receiving contexts,24 sources and eight random output spans. This is a diagnostic comparison, not an independent intervention confirmation. Total corpus documents12480. The underlying H434 calibration corpus is disjoint from receiving H438 documents, but some frozen source-map discovery used earlier H434 contexts; validation is conditional on those maps.

Use the H441384 training /128 validation baseline split. Build uncentered training covariance C from x/s, where x is RMS-normalized input and s its input RMS denominator, matching the scale of T*x/s. For each empirical/isotropic mixture0,.1,.5,.9,1, floor eigenvalues at1e-6 times training mean eigenvalue. Find top eigenvectors V of C^(1/2)*(sum T_r^T T_r)*C^(1/2). Decoder is C^(1/2)V, encoder V^T C^(-1/2); compile T_r*decoder. Select each rank128/256 mixture by held-out baseline analytic map reconstruction, with equal weighting of16 fixed source directions. No finite intervention outcomes enter selection.

The oblique encoder/decoder uses both map and input distribution, unlike orthogonal weight-only or activity-only projections. Calibration weighting also includes1/s, so gains cannot be attributed exclusively to obliqueness. Weight candidates at mixture0 provide an internal baseline. Preserve frozen simplified output fitting, exact radial and native retained contributions.

| Mixture | Rank | Validation map error fraction | Selected |
|---|---:|---:|---|
|0|128|0.081144|False|
|0|256|0.049426|False|
|0.1|128|0.060734|False|
|0.1|256|0.034152|False|
|0.5|128|0.056393|True|
|0.5|256|0.031531|True|
|0.9|128|0.062556|False|
|0.9|256|0.037225|False|
|1|128|0.065442|False|
|1|256|0.044152|False|

| Predictor | Learned finite KL recovery | Random mean recovery |
|---|---:|---:|
|dense_full_output|98.474%|95.887%|
|dense_simple_output|98.450%|95.430%|
|weight_128_simple_output|96.883%|93.821%|
|weight_256_simple_output|97.429%|94.380%|
|joint_128_simple_output|97.267%|94.214%|
|joint_256_simple_output|97.451%|94.413%|

| Frozen contrast | Estimate | Joint95% interval | Required lower | Pass |
|---|---:|---|---:|---|
|joint128_minus_weight128_learned|+0.003831|[-0.000499, +0.008161]|+0.005|False|
|joint128_minus_weight128_random_mean|+0.003937|[-0.000394, +0.008267]|+0.005|False|
|joint128_minus_dense_learned|-0.011838|[-0.016169, -0.007508]|-0.020|True|
|joint128_minus_dense_random_mean|-0.012158|[-0.016488, -0.007828]|-0.020|True|

Joint128-over-weight128 advantage requires0.5pp; joint128 noninferiority to dense/simple allows2pp. Four joint95% intervals use4000 paired source/context replicates and resampling of eight spans. Calibration/selection uncertainty is conditional, and rank256 is descriptive.

Audits: oblique compiled-versus-explicit projection response error1.11e-14; exact H452 reference replay error4.36e-17; solve residual3.76e-15; condition max1000.000000. All arrays finite and numerical gates pass, no exclusions.

Baseline validation selects50% empirical/50% isotropic covariance for both ranks. Rank128 analytic validation error falls8.114% at weight-only mixture0 to5.639% selected; rank256 falls4.943% to3.153%. Pure empirical covariance is worse than the selected mixture, consistent with limited calibration coverage or the value of shrinkage. These validation scores concern equal-weight source maps, not directly token probability effects.

Joint128 finite recovery is97.267% learned and94.214% randommean, versus98.450%/95.430% dense/simple and96.883%/93.821% weight128. Joint128 deficits are1.184pp learned and1.216pp random, joint95% intervals[−1.617,−0.751] and[−1.649,−0.783]pp. Both frozen2pp noninferiority gates pass. Measured losses remain; this is a practical tolerance result, not equality.

Joint128-over-weight128 gains are0.383pp learned and0.394pp random, joint95% intervals[−0.050,+0.816] and[−0.039,+0.827]pp. Both frozen0.5pp advantage gates fail, and intervals include zero. Consequently it is unjustified to say the new basis is conclusively superior simply because its noninferiority gates pass while H452’s weight gates did not: those joint bands use different contrast families. Rank256 joint97.451%/94.413% is only slightly above weight97.429%/94.380% on point estimates.

The key progress is a fully compiled rank128 input-response operation, with2,506,752 map-plus-encoder scalars (11.81% of dense map storage), whose measured behavioral loss stays inside the prespecified2pp tolerance on this reused panel. This moves beyond output-only fitting, but the unit remains conditional on16 source coordinates, native retained factors, exact radial correction and baseline state. It is neither a complete compressed MLP nor a discovered human semantic feature.

Next freeze the selected rank128/256 artifacts and the simplified output rule for confirmation on a new disjoint document panel and fresh sources/spans. H438’s512 receiving contexts are exhausted, so new corpus sampling and full-document hash exclusions are required before any independent-transfer claim. Preserve dense/simple, weight128 and joint128/256 controls; primary noninferiority should retain the2pp margin and separately quantify learned-versus-random and input-basis differences without conflating passed/failed gates. If compactness transfers, test source coverage beyond head6 and intervention magnitude before semantic naming. If it fails, examine distribution shift and calibration coverage rather than immediately adding rank.

Map-plus-encoder storage remains17*1152*rank scalars:2,506,752 at128 and5,013,504 at256. Decoder is compiled into maps and need not be stored at runtime. Baseline model, native retained terms, radial response, source-coordinate maps and output fitting costs are additional. No whole-model replacement, semantic ontology or speedup is established.
