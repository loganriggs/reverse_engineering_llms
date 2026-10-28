## H458 — Unprojected finite operation on unused receiving contexts

Freeze H457136 quadratic source-pair writes and exact rational formula, H453 joint256 context maps, and the source16 coordinate system. Test128 randomly chosen intervention-unused contexts from the384 remaining H454 corpus documents, with24 fresh natural/Gaussian source draws. Baselines were collected previously; no new corpus documents (total12992). The source donor pool remains fixed. There is no output span, coefficient fitting, Fisher solve or token-probability objective inside prediction; probabilities are used only to evaluate effects.

Compare unprojected dense source16 law, unprojected joint256 law and retained128-factor-only response at signed amplitudes. Both finite laws use the same exact baseline omitted write, source quadratic term and scalar RMS denominator. Exact retained responses use the original source h while omitted predictions use the frozen16-coordinate reconstruction hhat. The native full-source intervention is the target, so dense-law errors measure source restriction and compact-minus-dense comparisons isolate context-map approximation within this predictor.

| Amplitude | Dense finite KL recovery | Compact finite KL recovery | Retained-only recovery | Compact raw-response recovery |
|---|---:|---:|---:|---:|
|-2|99.332%|98.222%|84.106%|99.735%|
|-1|99.059%|97.512%|79.005%|99.734%|
|-0.5|99.033%|97.488%|78.478%|99.735%|
|-0.25|99.044%|97.529%|78.631%|99.736%|
|0.25|99.076%|97.625%|79.099%|99.738%|
|0.5|99.092%|97.669%|79.323%|99.739%|
|1|99.121%|97.741%|79.686%|99.742%|
|2|99.216%|97.860%|79.874%|99.748%|

| Frozen contrast | Estimate | Joint95% interval | Required lower | Pass |
|---|---:|---|---:|---|
|compact_minus_dense_negative2|-0.011094|[-0.019537, -0.002651]|-0.02|True|
|compact_minus_dense_positive2|-0.013561|[-0.022004, -0.005119]|-0.02|False|
|dense_recovery_negative2|+0.993317|[+0.984875, +1.001760]|+0.98|True|
|dense_recovery_positive2|+0.992158|[+0.983715, +1.000601]|+0.98|True|

Primary compact-minus-dense noninferiority allows2pp at−2 and+2; dense absolute recovery must exceed98% at each. Four joint95% max-absolute-deviation intervals use4000 paired source/document resamples. No random-span population remains because the predictor has no output projection. Other amplitudes, retained-only and raw-energy results are descriptive. Calibration uncertainty is conditional, not resampled.

Audits:128 receiving sites are distinct and disjoint from128 earlier intervention sites. Rational formula versus double direct source-restricted intervention relative error2.62e-13; native float32 projected-source error2.82e-05, checked at four fixed contexts/all amplitudes. All numerical gates pass, arrays finite, frozen artifact hashes checked and no exclusions.

The dense frozen source16 finite law passes both98% absolute-recovery gates:99.332% at−2 and99.216% at+2, joint95% intervals[98.487,100.176]% and[98.371,100.060]%. The symmetric intervals are not clipped; upper endpoints above100% do not imply achievable recovery above100%. This confirms strong prediction for these source draws and receiving contexts without output fitting.

Compact joint256 recovery is98.222% at−2 and97.860% at+2. Compact-minus-dense losses are1.109pp and1.356pp, joint95% intervals[−1.954,−0.265] and[−2.200,−0.512]pp. Negative2 passes the frozen2pp noninferiority gate; positive2 fails because its lower bound crosses−2pp. Therefore practical compactness is only partially confirmed, despite both point losses being inside2pp. No alternative margin or favorable amplitude is substituted.

Across all signed amplitudes compact recovery ranges97.488–98.222%, versus dense99.033–99.332% and retained-only78.478–84.106%. The fixed rational correction adds substantial point recovery beyond the retained native core. Curves are not monotonic in amplitude, and cross-panel changes cannot be assigned solely to receiving context because source draws also changed. All algebra/native checks pass; no output basis or probability-fit coefficients are used in prediction. The baseline model state/cache and source transform remain necessary.

This supports a concrete conditional computational object: a low-dimensional source-coordinate expansion interacting with256 context reads,136 quadratic source-pair writes and a scalar normalization law. It is operationally more explicit than a list of semantic labels, but its source restriction was built for head6 and does not establish a universal MLP unit. Accuracy alone also does not give those coordinates semantic meaning.

Next prioritize source coverage over further rank tuning. Apply the same frozen retained groups, source16 reconstruction, dense/compact rational laws to natural donor writes from all nine attention heads on matched receiving contexts. Include retained-only and report source-reconstruction error alongside raw/probability response fidelity. The head6 condition is an internal reference; other heads test the boundary of this proposed unit without refitting. If failure tracks missed source directions, the unit is source-conditional; if it transfers despite poor raw reconstruction, downstream equivalence matters. Use fresh donor draws from the fixed pool, avoid semantic claims from mere head identity, and preserve the current failed positive2 compactness gate. A conditional runtime benchmark remains optional and must include cache/setup accounting; broader semantic and cross-layer tests remain open.

Scope/cost: compact maps+encoder5,013,504 scalars and quadratic writes156,672, plus source geometry/transforms and retained-factor weights. The native baseline forward pass and cached omitted write are required. No per-edit full-vocabulary fitting remains, but no runtime speedup has yet been measured. Confirmation applies to this layer/head/source family, not model-wide semantic units or intact-model replacement.
