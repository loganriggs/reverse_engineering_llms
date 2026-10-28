## H452 — Input-coordinate compression of compiled source maps

Shift from output-fit simplification to the dense operation supplying its analytic response. Evaluate all96 remaining intervention-unused contexts in the H438512-document panel, with24 fresh source draws and eight fresh random output spans. All512 panel contexts have now been intervention-tested across experiments; no new corpus documents (total12480). Source donor calibration remains fixed.

For the16 fixed source-direction maps T_r, weight input directions are eigenvectors of sum_r T_r^T T_r. Activity input directions are eigenvectors of the uncentered covariance of RMS-normalized baseline inputs from512 H434 documents. Full-document hashes verify those calibration documents are disjoint from H438. No intervention target guides basis choice. Both bases are tested at32/64/128/256 directions without selection.

Compile each map as T_r B and compute its linear response from B^T x. This changes the input information available to the analytic correction, rather than merely fitting the dense answer into a smaller output span. Keep input RMS denominator, native retained response and exact radial correction uncompressed. Output fitting uses the frozen top128 constraints, shared tail mean, constant isotropic tail and receiving condition-cap1000 rule. Dense/simple and dense/full-output references separate source-map approximation from output-rule approximation.

| Predictor | Learned finite KL recovery | Random mean recovery | Linear-response energy coverage |
|---|---:|---:|---:|
|dense_full_output|98.474%|95.887%|reference|
|dense_simple_output|98.450%|95.430%|reference|
|weight_32_simple_output|95.378%|92.546%|87.667%|
|weight_64_simple_output|96.052%|93.074%|90.058%|
|weight_128_simple_output|96.883%|93.821%|92.573%|
|weight_256_simple_output|97.429%|94.380%|95.400%|
|activity_32_simple_output|95.173%|92.409%|87.794%|
|activity_64_simple_output|95.771%|92.939%|89.462%|
|activity_128_simple_output|95.973%|93.132%|91.037%|
|activity_256_simple_output|96.495%|93.614%|92.782%|

| Frozen rank128 contrast | Estimate | Joint95% interval | Lower > −.02 |
|---|---:|---|---|
|activity128_minus_dense_learned|-0.024773|[-0.035357, -0.014189]|False|
|activity128_minus_dense_random_mean|-0.022981|[-0.033565, -0.012397]|False|
|weight128_minus_dense_learned|-0.015669|[-0.026253, -0.005085]|False|
|weight128_minus_dense_random_mean|-0.016095|[-0.026679, -0.005511]|False|

Rank128 noninferiority margin is2pp, specified before outcomes. Joint95% bands use4000 paired source/receiving-context bootstrap draws and eight-span resampling. Rank curves and basis comparisons are descriptive outside these four gates. Calibration and frozen source-map uncertainty are conditional, not resampled.

Audits: compiled T*B response versus dense T*(B*B^T*x) relative error1.48e-15; solve residual3.76e-15; maximum regularized condition number1000.000000. All arrays finite and numerical gates pass. Recipient and calibration disjointness checked; no exclusions.

All four frozen rank128 noninferiority gates fail. Weight128 loses1.567pp learned and1.609pp randommean versus dense/simple, joint95% intervals[−2.625,−0.509] and[−2.668,−0.551]pp. Activity128 loses2.477pp learned and2.298pp random, intervals[−3.536,−1.419] and[−3.356,−1.240]pp. Both weight point losses are inside the2pp practical margin, but their lower bounds are not. These results demonstrate measured losses, not proof that every rank128 architecture must fail.

Dense/simple recovery98.450% learned/95.430% random is close on point estimates to dense/full98.474%/95.887%; this descriptive comparison is not a new frozen output-rule equivalence test. Weight input rank32/64/128/256 gives95.378/96.052/96.883/97.429% learned and92.546/93.074/93.821/94.380% random. Activity input ranks give95.173/95.771/95.973/96.495% learned and92.409/92.939/93.132/93.614% random. Weight-selected inputs outperform activity-selected inputs on these aggregate points; that contrast was not one of the primary gates.

Linear-response energy coverage increases from87.67% to95.40% across weight ranks32→256, versus87.79%→92.78% activity. Maximum correction norm ratio remains at most1.206 across conditions. The error trend is therefore consistent with loss of response information rather than the extreme extrapolation responsible for the earlier64-token collapse. Top activity directions alone do not efficiently capture the relevant map action. A rank256 weight variant uses5,013,504 map-plus-basis scalars, about23.6% of the21,233,664 dense-map scalars, at an approximately1pp behavioral recovery cost on this panel. This is conditional map-storage compression, not overall model compression or validated speedup.

Next distinguish a poor input-basis objective from intrinsically broad context dependence. Weight-only ranking ignores the input distribution; activity-only ranking ignores the map. Construct a joint reduced-rank regression basis from the stacked source maps times a regularized calibration covariance square root, with a matching inverse-root encoder. Freeze rank128 and256, select covariance regularization using held-out baseline analytic-map prediction only, and then test the frozen model against both current input bases. Ensure all comparisons have matched input rank and count oblique encoder/map storage. The next independent receiving test requires a new corpus panel: all512 H438 contexts have now been intervention-evaluated, so none may be labeled unused. Semantic and cross-layer scope remain open; the immediate question is whether a genuinely compact input-conditioned operation exists beyond convenient output fitting.

Storage: compiled maps plus shared input basis require17*1152*rank scalars:626,688 at32;1,253,376 at64;2,506,752 at128;5,013,504 at256. Dense16 maps require21,233,664 scalars. Counts exclude the baseline model, retained factors, source-coordinate transforms, radial correction, output basis and fitting computation. Compiled files are saved for every variant. No inference speedup is established; this is a conditional intervention predictor, not intact-model replacement or semantic ontology.
