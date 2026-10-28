## H455 — Signed amplitude separates response-law and compression errors

Reuse H454128 receiving contexts,24 sources and eight random output spans. Freeze dense/joint128/joint256 maps and simplified output-fitting rule. For source h, edit the MLP input as pre−alpha*h at alpha=−2,−1,−.5,−.25,.25,.5,1,2. Recompute exact native full and retained responses for each amplitude, keeping the direct residual path fixed. Scale the baseline analytic linear/radial correction and its fitted projection by alpha. No finite response is used to fit coefficients. Total corpus documents12992; this is a reused-panel mechanism diagnostic.

| Amplitude | Dense learned | Joint128 learned | Joint256 learned | Dense random mean | Joint128 random mean | Joint256 random mean |
|---|---:|---:|---:|---:|---:|---:|
|-2|96.224%|94.651%|95.356%|93.995%|92.434%|93.135%|
|-1|97.840%|96.280%|96.913%|95.507%|93.929%|94.568%|
|-0.5|98.321%|96.751%|97.368%|95.912%|94.321%|94.944%|
|-0.25|98.440%|96.865%|97.487%|95.979%|94.384%|95.010%|
|0.25|98.386%|96.796%|97.454%|95.812%|94.204%|94.859%|
|0.5|98.205%|96.600%|97.288%|95.588%|93.967%|94.647%|
|1|97.567%|95.919%|96.687%|94.942%|93.290%|94.036%|
|2|95.717%|94.024%|94.927%|93.301%|91.623%|92.476%|

| Frozen contrast | Estimate | Joint95% interval | Requirement | Pass |
|---|---:|---|---|---|
|dense_small_minus_large_learned|+0.026689|[+0.000539, +0.052838]|lower > .005|False|
|dense_small_minus_large_random|+0.025112|[-0.001037, +0.051262]|lower > .005|False|
|compression_gap_change_learned|+0.001417|[-0.024733, +0.027567]|whole interval inside [−.01,+.01]|False|
|compression_gap_change_random|+0.001290|[-0.024860, +0.027439]|whole interval inside [−.01,+.01]|False|

Primary dense deterioration contrasts recovery at+.25 minus+2. Compression-gap contrasts compare joint256-minus-dense at+2 versus+.25, with an equivalence margin1pp in either direction. Joint95% max-absolute-deviation bands use4000 paired source/context draws with eight-span resampling. Negative amplitudes and joint128 curves are descriptive controls. Different amplitudes have different full-intervention effect denominators; normalized recovery changes are not raw error magnitudes.

Audits: unit-amplitude target replay relative error0; full H454 unit-score replay max absolute error2.76e-16; solve residual2.25e-15; condition max1000.000000. Native float32 batching may affect replay; predeclared tolerances are1e-4 relative target and1e-6 absolute score. All gates pass; all arrays finite; no amplitude-dependent exclusions.

Dense/simple learned recovery decreases98.386%→95.717% from amplitude+.25 to+2; randommean95.812%→93.301%. Deteriorations2.669pp/2.511pp have joint95% intervals[0.054,5.284] and[−0.104,5.126]pp, so neither clears the frozen0.5pp deterioration threshold. Learned excludes zero but not the practical margin; random includes zero. All four frozen gates fail.

Joint256-minus-dense loss is0.932pp learned/0.953pp random at+.25 and0.791pp/0.825pp at+2. Gap changes are+0.142pp/+0.129pp, with joint95% intervals[−2.473,+2.757] and[−2.486,+2.744]pp. The small point changes favor a relatively amplitude-stable compression cost, but the equivalence intervals do not fit inside±1pp. Therefore compression-gap stability is not established. The joint band is widened by uncertainty in dense deterioration; the prespecified band is retained.

Signed curves are consistent with curvature: dense learned recovery at−.25/−.5/−1/−2 is98.440/98.321/97.840/96.224%, versus98.386/98.205/97.567/95.717% at matching positive amplitudes. Random spans and both compact ranks show similar broad trends. These are descriptive asymmetries; no separate sign-effect test was frozen. Since each effect denominator changes with amplitude, these recovery curves are not direct polynomial coefficients or isolated curvature energies.

Analysis audit: an initial NumPy mixed advanced-indexing expression interchanged amplitude and method axes when extracting primary bootstrap inputs. It was caught by comparing the printed contrast with the recovery table, before report publication. Replace with explicit np.take on named axes and assert that primary points equal contrasts recomputed from unweighted pooled bootstrap-input arrays. The corrected analyzer generates the reported intervals; initial incorrect outputs are superseded. GPU experiment data were unaffected. Unit-amplitude replay reproduces H454 and all numerical gates pass.

The leading mechanistic account is that first-order response approximation worsens with larger edits, while input compression adds a roughly persistent error on this panel. Neither the frozen effect-size nor stability claims are confirmed. The dense reference still restricts sources to16 calibrated directions; even small-amplitude error need not be curvature. Next separate this source truncation from derivative order: compare exact full-source omitted first derivative, source16 first derivative, and full-source second-order omitted prediction under the same fixed output rule and signed amplitudes. The exact retained response remains unchanged. This tests whether the compact operation needs richer source coordinates or nonlinear response terms before changing its input rank. Reuse this panel for the mechanistic diagnostic, then confirm any selected extension independently.

Scope: this tests a baseline first-order conditional response law with exact retained nonlinear responses, not a wholly linear model. Input compression and16-direction source restriction remain distinct approximations. The experiment does not by itself identify semantic units or provide intact-model replacement.
