## H446 — Regularization and the 64-token constraint boundary

H445 found a sharp failure at64 selected tokens for a64-dimensional correction, despite accurate regularized linear solves. H446 freezes budgets32/64/128 and ridge multipliers1e-6/1e-4/1e-2, plus full vocabulary at1e-6. The penalty is multiplier times mean within-span Fisher eigenvalue. Reuse the same128 contexts,24 sources and learned/four random spans; no new corpus documents (total12480). Preserve exact radial and native union128 contributions. Fit only analytic baseline response, then score full finite KL.

| Tokens | Ridge | Learned KL recovery | Random mean KL recovery | Largest learned norm ratio | Largest random norm ratio |
|---|---:|---:|---:|---:|---:|
|full|1e-06|98.493%|96.231%|1.588|1.290|
|32|1e-06|95.567%|93.512%|1.443|1.567|
|32|0.0001|95.570%|93.515%|1.441|1.563|
|32|0.01|95.707%|93.713%|1.299|1.403|
|64|1e-06|44.927%|-262.272%|42.088|51.766|
|64|0.0001|84.332%|61.542%|7.788|8.459|
|64|0.01|96.643%|92.641%|1.957|2.018|
|128|1e-06|98.085%|94.949%|1.963|1.906|
|128|0.0001|98.091%|94.954%|1.916|1.830|
|128|0.01|98.088%|95.255%|1.351|1.489|

Norm ratio is projected linear-correction magnitude divided by unprojected analytic linear-response magnitude, before adding the retained and radial components.

| Frozen contrast | Estimate | Joint95% interval | Required lower | Pass |
|---|---:|---|---:|---|
|ridge_rescue_learned|+0.517160|[-4.469347, +5.503666]|+0.02|False|
|ridge_rescue_random_mean|+3.549131|[-1.437376, +8.535637]|+0.02|False|
|stabilized64_minus_full_learned|-0.018503|[-5.005009, +4.968004]|-0.02|False|
|stabilized64_minus_full_random_mean|-0.035901|[-5.022407, +4.950606]|-0.02|False|

Inference uses4000 paired source/context resamples with random-span resampling and one joint maximum-absolute-deviation band for all four contrasts. This can make noninferiority bounds wide when rescue contrasts vary strongly. Frozen gates are retained; descriptive rescue does not imply established equivalence.

| Top64 condition | Learned included / omitted Fisher error fractions | Random mean included / omitted fractions |
|---|---|---|
|ridge 1e-06|0.008638 / 0.441439|0.025009 / 2.494120|
|ridge 0.0001|0.005283 / 0.130018|0.006975 / 0.329647|
|ridge 0.01|0.005748 / 0.024857|0.006606 / 0.060113|

These error fractions use full-distribution centered derivatives and the full intervention Fisher-energy denominator. They partition the combined predictor’s error, not just the linear regression objective. Consequently even selected tokens can retain error from omitted components or conditional-versus-full centering.

Median numerical ranks (full,32,64,128; learned then four random spans), threshold1e-10 times maximum eigenvalue: [[64.0, 64.0, 64.0, 64.0, 64.0], [31.0, 31.0, 31.0, 31.0, 31.0], [63.0, 63.0, 63.0, 63.0, 63.0], [64.0, 64.0, 64.0, 64.0, 64.0]]. Centering k token coordinates bounds rank at k−1; spectra and ridge effective degrees of freedom are saved, not treated as semantic feature counts.

Replay max absolute error 1.73e-17; solve relative residual 2.59e-14; numerical gates {'replay': True, 'solve': True}. Frozen hashes and finiteness checked. No outcome-dependent exclusions or ridge selection.

Mechanistic interpretation: the observed64-token collapse is largely unstable extrapolation. Median numerical rank is63 versus64 correction coordinates. Tiny-ridge output norm ratios reach42.09 learned and51.77 random, while most full-centered local error lies on omitted tokens. Ridge.01 reduces maxima to1.96/2.02 and restores learned/randommean finite recovery from44.93%/−262.27% to96.64%/92.64%. The32- and128-token conditions are much less sensitive to the same penalty change. This nonmonotonic budget response is evidence about fitting geometry, not semantic feature counts.

All four frozen statistical gates fail: the joint interval is extremely wide, driven by variability of the unstable random-span rescue. This limits population claims despite the clear measured norm/error change on this panel. An exploratory concentration audit finds a single receiving context accounts for82.09% and71.15% of top64 finite error in two random spans (other spans20.05% and23.85%; learned22.02%). These contexts differ by span. No cases are removed, and the audit is not independent confirmation. Stabilized64 still has point deficits1.85pp learned/3.59pp random against full vocabulary; fitting stability alone does not establish sufficient output information.

Next freeze a spectrum-based regularization rule that uses baseline geometry only: raise the ridge floor until the regularized Gram condition number is at most1e3, with a minimum ridge1e-6 times mean eigenvalue. Test budgets48/64/80/128 alongside full vocabulary, recording norms and included/omitted errors. A condition-based rule tests whether the rank-boundary problem can be prevented without selecting ridge by finite outcomes. Use independent receiving interventions and additional random spans for confirmation before claiming transfer; preserve the dense analytic-response cost and native/radial contributions. The broader objective of a compact, interpretable computational unit remains unresolved.
