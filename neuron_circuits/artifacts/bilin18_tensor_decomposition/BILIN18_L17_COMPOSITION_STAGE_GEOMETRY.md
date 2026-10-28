

## H397 — Where does apparent additivity arise?

Repeat the exact H396 sources and interventions, measuring the error of summing single-source rational laws at three stages. Same512 reused documents, three head pairs,13 actual/control conditions and four signed amplitude pairs. No coefficients are fitted to joint finite targets. The direct residual remains fixed.

Raw stage: error=||predicted MLP change−native MLP change||², effect=||native MLP change||². Final-normalized stage: error=||RMS(state+predicted change)−RMS(state+native change)||², effect=||RMS(state+native change)−RMS(state)||². Native stage: KL(target||prediction) and KL(target||intact), reused exactly from H396 and audited on the first four fixed sites of every condition/amplitude. Each relative error is pooled error divided by pooled effect, not a mean of site ratios. Squared norms and KL are different metrics; comparing their dimensionless relative-error ratios describes metric-dependent amplification, not a causal attribution or a comparison of absolute units.

Common-scale diagnostic: divide each raw squared error by the intact raw final-state squared norm at that site, then average. This denominator is shared by actual and control conditions, unlike intervention-relative recovery. Report log(control/actual) for this diagnostic; negative values mean smaller control raw error at that common scale. It still weights sites by inverse intact-state norm and is not an effect-size-matched causal estimate.

Frozen gates: {'cached_kl_replay': True, 'raw_unit99': True, 'normalization_suppresses': False, 'readout_suppresses': False}. Cached KL audit tolerance1e-10. Raw unit99 requires all18 simultaneous upper raw relative-error bounds<.01. Normalization/readout suppression each require all36 corresponding log stage-ratio upper bounds<0, using one joint72 band over both transitions. Four thousand paired document/control-draw resamples preserve site, pair, amplitude and stage dependence. The common-scale contrast has a separate joint24 band. Gates and metrics were frozen before stage outcomes.

|Pair|Family|Amplitudes|Raw error/effect|After RMS|Native KL|
|---|---|---|---:|---:|---:|
|[0, 1]|actual|[1, 1]|0.000236805|0.000186947|0.00053814|
|[0, 1]|actual|[1, -1]|0.000283413|0.000215184|0.000816965|
|[0, 1]|actual|[4, 4]|0.00283601|0.00228504|0.00766814|
|[0, 1]|actual|[4, -4]|0.00332968|0.00315114|0.00729666|
|[0, 1]|moment|[1, 1]|0.000209844|0.000181572|0.000522185|
|[0, 1]|moment|[1, -1]|0.000224775|0.000209545|0.000502484|
|[0, 1]|moment|[4, 4]|0.00400276|0.00303117|0.0110239|
|[0, 1]|moment|[4, -4]|0.00394169|0.00398446|0.00606064|
|[0, 1]|donor|[1, 1]|0.000216077|0.000188099|0.000525424|
|[0, 1]|donor|[1, -1]|0.000232227|0.000217949|0.000500622|
|[0, 1]|donor|[4, 4]|0.00433358|0.00322809|0.0130997|
|[0, 1]|donor|[4, -4]|0.00410005|0.00416587|0.0064811|
|[0, 6]|actual|[1, 1]|9.93909e-05|6.20983e-05|0.000392027|
|[0, 6]|actual|[1, -1]|9.08559e-05|5.3183e-05|0.000355759|
|[0, 6]|actual|[4, 4]|0.00114023|0.000801917|0.00522071|
|[0, 6]|actual|[4, -4]|0.00114288|0.000679936|0.00430602|
|[0, 6]|moment|[1, 1]|7.17191e-05|6.11198e-05|0.000116996|
|[0, 6]|moment|[1, -1]|7.44514e-05|6.12208e-05|9.44183e-05|
|[0, 6]|moment|[4, 4]|0.00109924|0.00108567|0.00212591|
|[0, 6]|moment|[4, -4]|0.001258|0.000963833|0.00135576|
|[0, 6]|donor|[1, 1]|6.60436e-05|5.61611e-05|0.000102337|
|[0, 6]|donor|[1, -1]|6.8347e-05|5.55712e-05|7.99057e-05|
|[0, 6]|donor|[4, 4]|0.00101399|0.000998219|0.00194022|
|[0, 6]|donor|[4, -4]|0.00112776|0.000859933|0.00114175|
|[1, 6]|actual|[1, 1]|0.00147285|0.00149454|0.00277844|
|[1, 6]|actual|[1, -1]|0.00126088|0.00138826|0.00274741|
|[1, 6]|actual|[4, 4]|0.0143907|0.0206189|0.0363169|
|[1, 6]|actual|[4, -4]|0.0184706|0.0244072|0.0311801|
|[1, 6]|moment|[1, 1]|0.00138216|0.00252474|0.00150336|
|[1, 6]|moment|[1, -1]|0.00102413|0.00141862|0.0011743|
|[1, 6]|moment|[4, 4]|0.0224801|0.0431158|0.0242641|
|[1, 6]|moment|[4, -4]|0.0136597|0.018012|0.0145627|
|[1, 6]|donor|[1, 1]|0.00142086|0.00282081|0.00145644|
|[1, 6]|donor|[1, -1]|0.00110161|0.00154499|0.00113967|
|[1, 6]|donor|[4, 4]|0.0231662|0.0470155|0.0246687|
|[1, 6]|donor|[4, -4]|0.0143634|0.0174071|0.0138678|

Audit: {'cached_kl_max_absolute_replay_error': 6.938893903907228e-17, 'saved_geometry_max_absolute_replay': 0.0, 'joint_identity_max_float32_relative_error': 0.0007051649736240506}. Saved-array validation: {'relative_error_replay_max_absolute': 3.469446951953614e-18, 'cached_kl_arrays_identical': True, 'log_stage_telescope_max_absolute_error': 4.440892098500626e-16}. Full sitewise stage errors/effects, common-scaled errors and bootstrap arrays are retained. Native KL values are retained without clipping. Only four fixed sites per condition/amplitude are independently rescored through the unembedding; this audit does not constitute a fresh full-panel KL recomputation.

Outcome: cached native-KL audit and raw unit99 gates pass. Both universal stage-suppression gates fail. The cache audit agrees to6.94e-17 and regenerated single-source geometry agrees exactly. All raw/unit pooled squared-error ratios lie between0.00006604 and0.00147285 (0.006604–0.147285%); the largest simultaneous upper bound is0.00168628 (0.168628%). Thus approximate additivity is already present before final RMS and readout for ordinary signed removals, including matched control sources. These are squared-error fractions: their square roots range approximately0.81–3.84%, so they must not be described as sub0.15% vector-norm error.

Final normalization changes relative error by factors0.585–2.029. Under the joint72 band, five of36 normalization contrasts significantly amplify relative error, three suppress it, and28 remain unresolved. The native readout changes relative error by factors0.516–6.689:25 significantly amplify, four suppress, and seven remain unresolved. These are dimensionless changes between differently defined metrics, not an additive causal decomposition. Neither stage is a universal error suppressor. In particular, for actual heads0/6, the native readout amplifies relative error by roughly6.3–6.7× across the four amplitude conditions.

Heads1/6 illustrate why source and sign matter. At(4,4), actual raw squared-error fraction is1.439%, after RMS2.062%, and native KL3.632%. Moment controls progress2.248%→4.312%→2.426%; donor controls2.317%→4.702%→2.467%. Controls therefore have larger raw and normalized relative errors, yet smaller native relative errors. A ranking based only on native recovery would reverse the raw-space ranking. This is a readout-dependent comparison, not evidence that control source computations are inherently more independent.

The common-scale raw-error diagnostic removes the intervention-effect denominator. For heads1/6 at(4,4), moment/actual error ratio is2.261, joint interval[1.887,2.710]; donor/actual2.682,[2.238,3.215]. At(4,−4), the ordering reverses: moment/actual0.721,[0.602,0.864], donor/actual0.811,[0.676,0.972]. Across all24 common-scale comparisons, eight significantly favor actual sources (controls have more error), five favor controls, and eleven are unresolved. No universal actual-source independence is supported. The sign reversal persists without dividing by each intervention's own effect size.

Hypothesis update: generic weak cross-source interaction at ordinary amplitudes is supported in raw space, not just in the output metric. Universal normalization/readout suppression is contradicted on this panel. Sign-dependent coupling and metric-dependent ordering remain important. The architecture's exact single-direction rational law is useful, but neither its algebraic compactness nor high native recovery identifies semantic units. These outcomes remain local to L17, three selected head pairs, reused documents and the stated control construction.

Next discriminating experiment: test the leading generic explanation against source organization by expanding to all36 head pairs with a cheap raw-space interaction screen, retaining actual and matched moment controls, both signs and ordinary/larger amplitudes. Freeze a criterion for unusually weak or strong coupling relative to controls, account for multiple comparisons and source effect balance, and confirm selected pairs on fresh documents before detailed output scoring. This broadens source coverage instead of extrapolating the three selected pairs. A separate full-geometry matched control would be needed to isolate mutual-angle effects; current controls do not provide that isolation. Subsequent native-factor sparsity or cross-context transfer tests should target reproducible exceptional pairs, if any, rather than calling every locally accurate response law a discovered computational feature.
