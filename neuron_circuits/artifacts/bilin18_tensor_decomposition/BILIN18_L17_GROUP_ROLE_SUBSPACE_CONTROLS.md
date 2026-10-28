

## H414 — Separate actual source activations from head-subspace geometry

Reuse H413512 documents and frozen H410 factor groups. Actual controls replay H413. For heads4,6,8 form the128-dimensional output basis from that head's native attention output projection. Replace actual source vectors with six isotropic draws or six mean/covariance-based draws using the frozen H333 per-position source moments. The two families use paired Gaussian seeds. Each replacement is projected/scaled to preserve the actual source norm and its dot product with the recipient residual while remaining in the same head subspace. The recipient state/output metric and factor groups remain fixed.

The moment draw is geometry-matched after sampling; its final moments are not claimed to equal the fitted moments exactly. These are counterfactual vectors, not natural attention outputs. Matching norm and angle retains some recipient information and source-specific geometry; this intervention does not erase every possible content-related property.

Evaluate the same three statistics: head6 and head8 core/add cross divided by full Fisher energy, and head4 additions-minus-core finite native recovery. Pool numerator and denominator over sites/draws. Four thousand paired document/draw bootstrap resamples form one joint6 band for the two control families' role values, plus a separate joint6 band for their differences from actual. Role-preservation gates require all three original thresholds within each family (upper<−.05, lower>.05, lower>.1). Equivalence requires all three difference intervals wholly within±.05. Geometry errors must be<1e-5; actual replay<1e-9. No post-outcome threshold changes.

|Family|Role statistic|Actual|Control|Control joint6 interval|Control−actual joint6 interval|
|---|---|---:|---:|---|---|
|isotropic|head6_core_add_cross_over_full|-0.167567|0.044230|[0.0009827136541564266, 0.08747792362857537]|[0.16613140151813882, 0.2574627319977145]|
|isotropic|head8_core_add_cross_over_full|0.115869|0.051461|[0.008213743196379159, 0.09470895317079811]|[-0.11007342326553579, -0.01874209278596011]|
|isotropic|head4_additions_minus_core_finite_recovery|0.433501|0.391332|[0.34808438727335506, 0.434579597247774]|[-0.08783482913322602, 0.003496501346349677]|
|moment|head6_core_add_cross_over_full|-0.167567|0.060599|[0.017351487503847705, 0.10384669747826666]|[0.18250017536783011, 0.2738315058474058]|
|moment|head8_core_add_cross_over_full|0.115869|0.140827|[0.09757943782559128, 0.18407464780001023]|[-0.020707728636323676, 0.070623601843252]|
|moment|head4_additions_minus_core_finite_recovery|0.433501|0.398457|[0.3552094129408414, 0.44170462291526036]|[-0.08070980346573967, 0.010621527013836028]|

Gates: {'geometry': True, 'actual_replay': True, 'isotropic_roles': False, 'isotropic_equivalence05': False, 'moment_roles': False, 'moment_equivalence05': False}.

|Family|Head6 Fisher energy/control-to-actual|Head8 Fisher energy/control-to-actual|Head4 finite effect/control-to-actual|
|---|---:|---:|---:|
|isotropic|1.552962|0.245982|0.415189|
|moment|2.358137|0.739630|0.668217|

Absolute-effect ratios are descriptive; equality of normalized role metrics would not imply identical effect magnitudes. Audits: {'geometry': {'relative_norm': 4.280033202639984e-07, 'normalized_residual_dot': 2.8290175536012854e-07, 'relative_span_residual': 5.11002307493113e-07, 'native_relative_span_residual': 5.757932513006381e-07}, 'actual_arrays_replay_max_absolute': 4.85722573273506e-16}. Independent saved-array checks: {'pooled_role_stats_replay': True, 'control_differences_replay': True, 'all_arrays_finite': True}. All draw/site energies and KLs, random seeds and frozen source hashes are retained.

Outcome: geometry and actual replay pass. Both all-three-role preservation gates and both all-three-metric equivalence gates fail. The failure is mechanistically informative: actual head6 cross/full is−0.167567, while isotropic replacements give+0.044230,[0.000983,0.087478] and moment-based replacements give+0.060599,[0.017351,0.103847], using the simultaneous control-value band. Control-minus-actual head6 differences are+0.211797,[0.166131,0.257463] and+0.228166,[0.182500,0.273832], in the separate simultaneous difference band. Cancellation is lost under both replacements; preserving the head subspace, source norm and recipient projection is insufficient to reproduce it.

This is not merely a smaller response washing out the cross term: head6 full Fisher energy increases to1.553× actual for isotropic draws and2.358× for moment draws. Moreover, changing a positive denominator cannot explain the negative-to-positive cross-term sign change. Actual source direction therefore matters for this grouping in this recipient-conditioned metric. That statement does not establish semantics, nor distinguish context-specific pairing from non-Gaussian source-direction structure.

Other roles are less activation-specific. Head8 cross/full is+0.051461,[0.008214,0.094709] for isotropic draws and+0.140827,[0.097579,0.184075] for moment draws, compared with actual+0.115869. The moment control individually retains the+.05 role threshold; isotropic control retains positive sign but does not clear+.05. Its difference from actual is−0.064408,[−0.110073,−0.018742], whereas the moment difference+0.024958,[−0.020708,0.070624] includes zero but does not establish±.05 equivalence.

Head4 additions-over-core finite recovery remains large for both controls: isotropic0.391332,[0.348084,0.434580] and moment0.398457,[0.355209,0.441705], each individually clearing+.1, versus actual0.433501. Differences from actual are−0.042169,[−0.087835,0.003497] and−0.035044,[−0.080710,0.010622]. Thus the role persists, but magnitude equivalence is not established. Head4 absolute native finite effect falls to0.415×/0.668× actual under isotropic/moment controls; similar normalized usefulness does not imply the same computation magnitude. Head8 full Fisher energy is0.246×/0.740× actual respectively.

All geometric discrepancies are below5.76e-7, including original native-source subspace residual; actual role input arrays replay H413 within4.86e-16 absolute error. Means/covariances are frozen from H333 references; no controls were fitted to H413. Angle matching changes the sampled distribution, so these are not exact final mean/covariance matches. Paired confidence intervals condition on the frozen moment estimates, finite draw family and recipient corpus.

Hypothesis update: #12 (head-subspace geometry alone explains every role) is insufficient for head6 cancellation, although it helps account for head4's broad added-factor usefulness and head8 alignment. #5 remains supported with an activation-dependent scope. The leading alternatives now are (a) non-Gaussian/higher-order marginal source directions lost by Gaussian sampling, and (b) the pairing of source direction with recipient context. The present controls change both. Recipient norm/projection information was deliberately retained, so do not claim that every contextual property was randomized away. No semantic feature interpretation is warranted.

Next discriminating experiment: use actual head outputs from a disjoint older document panel as donors, preserving token position. Map donors to these recipients with six frozen permutations, then apply the same within-head norm/residual-angle matching. Keep the native higher-order direction distribution before the matching transform while breaking the particular source-recipient pairing. Test all three role statistics, with head6 donor-minus-actual and donor-versus-moment contrasts frozen before outcomes. If natural donors restore cancellation, Gaussian marginal misspecification becomes plausible; if cancellation still disappears, recipient pairing becomes the stronger explanation. Use the fixed donor pool explicitly and report uncertainty conditional on that pool rather than claiming a full donor-population confidence interval. No new factor selection is needed.

No fresh documents were consumed. Future fresh panels must still exclude8896 completed-panel documents. The broad research goal remains open, and no experiment processes remain running.
