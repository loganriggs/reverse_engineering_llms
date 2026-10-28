## H456 — Source coverage versus derivative order

Reuse H454/H455128 contexts,24 sources,eight output spans and signed amplitudes. Compare source16 first-order prediction with full-source first order and full-source first-plus-second order. Keep the same simplified baseline output fit, exact retained finite response and receiving state. First-order radial correction is kept outside the output projection for its corresponding source; second-order omitted response is projected through the same baseline rule. Full-source unprojected first/second controls separate analytic approximation from output-fit loss. No finite target enters fitting; total12992 documents, no fresh confirmation.

For v=−h and RMS scale s, normalized-input expansion is x+alpha*u1+alpha^2*u2, where u1=DRMS(pre,v) and u2=pre*(−mean(v^2)/(2s^3)+3mean(pre*v)^2/(2s^5))−v*mean(pre*v)/s^3. The omitted bilinear second-order coefficient is J_B(x)*u2+B(u1,u1), excluding retained factors. This coefficient already includes the Taylor1/2 convention. The full first derivative decomposes into linear-input and radial terms, audited separately.

| Amplitude | Source16 learned | Full first learned | Full second learned | Source16 random | Full first random | Full second random | Unprojected first | Unprojected second |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|-2|96.224%|96.659%|98.281%|93.995%|94.394%|95.169%|97.610%|99.172%|
|-1|97.840%|98.246%|98.843%|95.507%|95.880%|96.159%|99.314%|99.958%|
|-0.5|98.321%|98.710%|98.879%|95.912%|96.277%|96.343%|99.817%|99.998%|
|-0.25|98.440%|98.824%|98.869%|95.979%|96.346%|96.358%|99.953%|100.000%|
|0.25|98.386%|98.772%|98.813%|95.812%|96.193%|96.222%|99.950%|100.000%|
|0.5|98.205%|98.599%|98.773%|95.588%|95.980%|96.080%|99.796%|99.998%|
|1|97.567%|97.983%|98.692%|94.942%|95.359%|95.739%|99.177%|99.970%|
|2|95.717%|96.113%|98.330%|93.301%|93.702%|94.785%|97.171%|99.547%|

| Frozen contrast | Estimate | Joint95% interval | Lower > .005 |
|---|---:|---|---|
|fullsource_minus_source16_small_learned|+0.003861|[-0.014119, +0.021842]|False|
|fullsource_minus_source16_small_random|+0.003805|[-0.014176, +0.021786]|False|
|second_minus_first_large_learned|+0.022169|[+0.004188, +0.040150]|False|
|second_minus_first_large_random|+0.010832|[-0.007148, +0.028813]|False|

Primary source-coverage gains are evaluated at+.25; second-order gains at+2. Four joint95% intervals use4000 paired source/context draws, eight-span resampling and a max-absolute-deviation band. Each practical advantage threshold is0.5pp. Signed and unprojected curves are descriptive. Explicit-axis bootstrap extraction includes an unweighted point-reconstruction assertion following H455’s indexing audit.

Audits: full-source first-order linear/radial identity relative error9.48e-15; central second finite-difference relative error6.5e-07 at four fixed contexts, step.001; H455 source16 all-amplitude replay absolute error0; solve residual1.72e-15. All numerical gates pass; no cases excluded.

All four frozen practical-advantage gates fail. At+.25, full-source first-order improves over source16 by0.386pp learned and0.381pp randommean, joint95% intervals[−1.412,+2.184] and[−1.418,+2.179]pp. Source-direction truncation has a modest point cost here, but these bands do not establish its magnitude or absence.

At+2, full-source first/second recovery is96.113%/98.330% learned and93.702%/94.785% random. Second-order gains are2.217pp learned and1.083pp random, joint95%[0.419,4.015] and[−0.715,2.881]pp. Learned excludes zero but narrowly misses the frozen0.5pp lower-bound requirement. Random includes zero. No failed threshold is relabeled. Second-order gains are much smaller at+.25 (about0.041pp learned/0.029pp random), consistent on point estimates with amplitude-dependent curvature rather than a fixed source-coverage deficit.

Unprojected full-source controls are substantially more accurate. At+.25, direct first/second recovery is99.950%/99.9999%; at+1,99.177%/99.970%; at+2,97.171%/99.547%. The projected learned second-order score at+2 is98.330%, random94.785%. Thus accurate input-derived nonlinear prediction does not eliminate output-projection loss. The unprojected curves are descriptive controls, not additional preregistered comparisons. At−2 direct second recovery99.172% versus99.547% at+2 also cautions against assuming perfect sign symmetry or unlimited Taylor validity.

All numerical gates pass, including the full-source linear/radial identity, independently evaluated central second finite difference at four fixed contexts, and source16 all-amplitude replay of H455. The full-source derivative computation uses the original omitted weights and is not compact. The current leading account has at least three distinct approximations: source-coordinate restriction, input-map compression and output projection, plus Taylor truncation for finite edits. Their errors must not be conflated with semantic feature counts.

Next build an exact finite response law inside the frozen16-dimensional source subspace, rather than merely adding more Taylor terms. For projected source hhat=Hc, the bilinear numerator has baseline write, linear T_hhat*x and a quadratic source-only term B(hhat,hhat); RMS supplies an explicit scalar denominator. The quadratic source term can be compiled into136 symmetric source-pair writes (16*17/2), about156672 output coefficients. Compare exact projected-source finite replay, dense linear maps and rank256 linear maps, both before and after output projection, across the existing signed amplitude panel. Audit against direct native projected-source interventions before drawing scientific conclusions. This tests whether the nonlinear response itself admits a compact executable operation; baseline omitted write, retained response and source/input information costs remain explicit. Independent confirmation must follow any selected extension.

Cost/scope: full-source first/second derivatives evaluate original dense MLP weights and4480 omitted factors. This is a mechanistic diagnostic, not yet a compact replacement. The output fit remains context-dependent, and first-order radial terms remain explicit. Strong predictive accuracy would establish an input-derived response law within this setup, not semantic labels or whole-model sufficiency.
