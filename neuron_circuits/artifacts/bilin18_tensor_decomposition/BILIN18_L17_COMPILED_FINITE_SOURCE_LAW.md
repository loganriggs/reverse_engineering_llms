## H457 — Compiled finite law inside the source subspace

Reuse H454/H455128 contexts,24 sources,eight output spans and signed amplitudes. Reconstruct hhat=Hc using the frozen16 source coordinates. Compile136 symmetric source-pair writes for B_omit(hhat,hhat). Retain the source maps T_r and optionally replace their context-input action with frozen joint256 maps. No finite target is used to construct or fit the law; total12992 documents.

Let s²=mean(pre²)+epsilon, x=pre/s, d=mean((pre−alpha*hhat)²)+epsilon and M0=M_omit(x). The exact omitted response within this source subspace is ((s²−d)*M0−alpha*s*T_hhat*x+alpha²*B_omit(hhat,hhat))/d. The scalar denominator uses source-coordinate dot products and a16x16 source Gram matrix. This is an exact rational expression, not a Taylor truncation. The original source h still supplies the exact retained response; the omitted predictor uses hhat, so the combined predictor is not a single native intervention at hhat.

The baseline-normalization component (s²/d−1)*M0 remains exact outside output projection. Project the remaining linear-plus-quadratic numerator divided by d through the frozen simplified baseline output rule. Unprojected controls add both terms directly. Dense and joint256 variants differ only in the context-input linear maps. Gains versus first order combine quadratic numerator and exact normalization changes; this experiment does not separately attribute them.

| Amplitude | Dense learned | Joint256 learned | Dense random mean | Joint256 random mean | Unprojected dense | Unprojected joint256 |
|---|---:|---:|---:|---:|---:|---:|
|-2|98.503%|97.427%|95.684%|94.592%|99.449%|97.960%|
|-1|98.512%|97.496%|95.956%|94.917%|99.477%|98.043%|
|-0.5|98.507%|97.530%|96.021%|95.025%|99.484%|98.090%|
|-0.25|98.491%|97.532%|96.004%|95.027%|99.484%|98.103%|
|0.25|98.426%|97.490%|95.843%|94.888%|99.473%|98.097%|
|0.5|98.384%|97.452%|95.709%|94.754%|99.463%|98.080%|
|1|98.328%|97.391%|95.437%|94.474%|99.440%|98.045%|
|2|98.384%|97.475%|95.007%|94.057%|99.433%|98.128%|

| Frozen +2 contrast | Estimate | Joint95% interval | Required lower | Pass |
|---|---:|---|---:|---|
|finite_minus_first_learned|+0.026669|[+0.004251, +0.049087]|+0.005|False|
|finite_minus_first_random|+0.017057|[-0.005361, +0.039475]|+0.005|False|
|compact_minus_dense_finite_learned|-0.009089|[-0.031507, +0.013330]|-0.020|False|
|compact_minus_dense_finite_random|-0.009497|[-0.031915, +0.012921]|-0.020|False|

Four joint95% intervals use4000 paired source/context replicates with eight-span resampling. Finite-dense versus source16-first requires0.5pp advantage; compact-finite versus dense-finite allows2pp loss. All use amplitude+2; other amplitudes and unprojected comparisons are descriptive.

Audits: compiled quadratic relative error8.43e-15; rational law versus direct double-precision source-subspace intervention1.77e-13; native float32 projected-source relative error2.74e-05 at four fixed contexts/all signed amplitudes; effect replay absolute error0; solve residual1.84e-15. All gates pass and saved arrays finite. No exclusions.

The exact finite-law audits pass:136 compiled quadratic source-pair writes reproduce direct bilinear evaluation, and the rational formula matches double and native source-restricted interventions. This verifies the algebra, not adequacy of the16-source approximation for arbitrary h. The finite predictor preserves the original source’s retained response and approximates its omitted response through hhat.

At+2, projected finite-dense recovery is98.384% learned/95.007% random, versus95.717%/93.301% source16 first order. Gains2.667pp/1.706pp have joint95% intervals[0.425,4.909] and[−0.536,3.947]pp; neither clears the frozen0.5pp advantage gate. Learned excludes zero but narrowly misses the practical threshold. Projected finite joint256 reaches97.475%/94.057%, losses0.909pp/0.950pp versus dense-finite, intervals[−3.151,+1.333] and[−3.191,+1.292]pp. Both2pp noninferiority gates fail. All four primary gates remain failed.

The unprojected controls suggest a simpler and more accurate operation. Dense source16 finite recovery stays99.43–99.48% across positive amplitudes and99.45–99.48% across negative amplitudes. Unprojected joint256 stays98.08–98.13% positive and97.96–98.10% negative. At+2, direct dense/compact is99.433%/98.128%, both above their corresponding learned and random projected scores. These controls were descriptive, not the frozen primary comparisons; they motivate independent testing, not a retrospective success claim. Dense residual error reflects source reconstruction and native precision rather than Taylor truncation. Compact error additionally reflects input-map approximation.

The candidate computational unit is now explicit:128 native retained responses plus an omitted-response operation comprising256 context reads,16 source coordinates,136 source-pair writes, a cached baseline omitted write and a scalar RMS denominator. It predicts finite edits without fitting coefficients against token probabilities if the unprojected variant is used. Context/source transforms and exact baseline information still carry essential structure; this is not a standalone compressed language model or semantic dictionary.

Next freeze the unprojected joint256 rational formula for confirmation on intervention-unused contexts from the H454 fresh512 panel (exclude its128 previously evaluated recipients). Use fresh source draws and compare exact native target, unprojected dense source16 law, unprojected compact law and retained-only response at signed amplitudes. Primary compact-minus-dense noninferiority should be fixed before outcomes, not chosen from these curves. This directly tests the simpler operation and eliminates output-span choice/metric fitting from prediction. Separately account for source16 error against the native target. A conditional runtime benchmark may follow confirmation, with baseline forward/cache construction costs reported explicitly; semantic and cross-head/layer transfer remain open.

Cost/scope: quadratic source writes add136*1152=156672 scalars; source Gram adds256 and source directions18432. Joint256 context maps+encoder add5,013,504. Baseline omitted write, exact native retained response, source-coordinate transform, baseline forward pass and output-fitting costs remain outside these counts. Exact source-subspace algebra is not proof that the restricted source space suffices for all interventions. No model-wide compression or semantic labeling claim is made.
