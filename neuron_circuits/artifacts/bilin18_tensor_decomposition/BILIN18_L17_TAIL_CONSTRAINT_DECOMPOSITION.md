## H448 — Within-tail variation versus between-group centering

Reused H447128 contexts,24 sources, learned span and eight random spans. No new confirmation or corpus documents; total12480. This diagnostic decomposes the full baseline probability-weighted regression objective into top128 within-group covariance, tail within-group covariance and the between-group mean contrast. It also shuffles probability weights among tail token identities, preserving the tail weight multiset and total mass while recomputing both regression matrix and analytic target cross-covariance coherently.

For head mass m and tail mass n=1−m, full covariance is m*Cov(head)+n*Cov(tail)+m*n*(mean_head−mean_tail)(mean_head−mean_tail)^T. The right-hand side obeys the analogous cross-covariance identity. Omitting the final rank-one term would silently change the full objective. The exact identity and H447 full/top replay are numerical gates.

All conditions use the baseline condition-number cap1000, except the explicit full-tiny reference. Thus comparisons hold the regularization rule fixed, but the selected penalty can change with each matrix. Within-only and between-only conditions are mechanistic ablations, not replacements with matched information. Native union128 and exact radial terms remain. A shared calibration surrogate is deferred until these exact components are understood.

| Condition | Learned finite KL recovery | Random mean finite KL recovery | Learned included / omitted local error fractions | Random included / omitted fractions |
|---|---:|---:|---|---|
|top128_condition|97.932%|95.294%|0.007523 / 0.012954|0.011993 / 0.034965|
|full_tiny|98.470%|96.447%|0.007547 / 0.006812|0.014316 / 0.020566|
|full_condition|98.440%|96.446%|0.007718 / 0.007008|0.014332 / 0.020566|
|top_plus_tail_within|98.414%|96.242%|0.007768 / 0.007267|0.014464 / 0.022486|
|top_plus_between|98.067%|95.345%|0.007416 / 0.011625|0.011986 / 0.034345|
|shuffled_tail_condition|98.388%|95.855%|0.007887 / 0.007297|0.016866 / 0.023791|

| Frozen contrast | Estimate | Joint95% interval | Lower > .005 |
|---|---:|---|---|
|full_minus_top_learned|+0.005089|[+0.002481, +0.007697]|False|
|full_minus_top_random_mean|+0.011523|[+0.008915, +0.014131]|True|
|full_minus_shuffle_learned|+0.000524|[-0.002084, +0.003132]|False|
|full_minus_shuffle_random_mean|+0.005917|[+0.003310, +0.008525]|False|

Joint95% intervals use4000 paired source/document bootstrap replicates, resampling eight random spans and a maximum-absolute-deviation band across four contrasts. The prespecified practical threshold is0.5 percentage points. Failed advantage gates are not evidence of equivalence. Intervention targets enter evaluation only, never the coefficient fit.

Audits: decomposition relative error2.74e-15; replay absolute error1.64e-16; solve relative residual5.85e-15. All numerical gates pass and all saved errors/norms are finite. No exclusions.

The full stable fit adds0.509pp learned and1.152pp random over stable top128. Joint95% intervals are[0.248,0.770] and[0.892,1.413]pp; only random clears the frozen0.5pp practical threshold. Both intervals exclude zero, but that is weaker than both passing the specified gate. Full tiny and full condition scores are nearly identical, so the remaining top128 deficit is not explained by the full-fit regularization rule alone on this panel.

Adding within-tail covariance/cross-covariance captures most of the point-estimate full-tail gain: learned98.414% and random96.242%, versus98.440%/96.446% full stable and97.932%/95.294% top only. Adding only the between-group contrast yields98.067%/95.345%. These ablations favor within-tail variation over a pure head-versus-tail mean explanation. Their relative attribution is descriptive; they are not separately preregistered inferential comparisons and re-solving changes coefficients and ridge magnitude.

Shuffling weights among tail identities retains98.388% learned and95.855% random recovery. Full-minus-shuffled is0.052pp learned (joint95%[-0.208,0.313]) and0.592pp random ([0.331,0.853]). Neither clears the0.5pp practical threshold; the random interval nevertheless excludes zero. Thus exact tail probability assignment has measurable signal in the random-span contrast, while much useful tail geometry survives reassignment. This is not equivalence and not a semantic-token interpretation. The control preserves every tail token, probability mass and weight multiset, jointly recomputing target and basis statistics; it destroys their original weight association.

Updated hypothesis ranking: (1) ordinary within-tail output geometry is a leading explanation of the residual top128 gap; (2) context-specific probability-to-token association adds a smaller tested contribution; (3) a pure between-group-centering account is weakened; (4) the top128 deficit is not solely conditioning; (5) semantic specificity remains unestablished. Next replace exact within-tail geometry with a reusable baseline-calibrated covariance, keeping target fitting coherent and preserving the exact between-group term for attribution. Compare isotropic and shared anisotropic controls, calibrated without these intervention outcomes, then test transfer. Count covariance storage and dense analytic-response construction explicitly. A successful shared approximation would support reusable output constraints rather than a distinct semantic axis for every fitted coordinate; failure would motivate preserving context-specific tail geometry.

Scope: this diagnoses a context-dependent coefficient rule inside a structured predictor. It does not establish semantic axes, a whole-model ontology or an inference speedup. Dense analytic maps, baseline forward pass and full tangent computations remain.
