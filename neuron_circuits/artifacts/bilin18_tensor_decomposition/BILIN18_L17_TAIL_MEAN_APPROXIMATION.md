## H451 — Shared versus zero tail mean

Reuse H450128 receiving contexts,24 sources and eight random spans; total12480 corpus documents, no new independent confirmation. Estimate a shared conditional tail logit-gradient mean from disjoint H444128 baseline states, then freeze it. Use the same H450 constant isotropic tail covariance trace. Preserve exact head constraints, receiving tail mass, output-RMS transport, native union128 and radial response.

Compare exact receiving tail mean, shared calibration mean, and zero tail mean. Recompute the between-group Gram matrix and analytic-response cross-covariance consistently from the head-minus-tail mean. Zero tail mean preserves a head-versus-zero rank-one term; it is distinct from the no-between control, which omits the term entirely. Full exact covariance and top-only conditions are references. All fits use condition cap1000.

| Condition | Learned finite KL recovery | Random mean recovery |
|---|---:|---:|
|full_condition|98.166%|95.744%|
|constant_exact_tail_mean|98.131%|95.599%|
|constant_shared_tail_mean|98.132%|95.444%|
|constant_zero_tail_mean|98.100%|95.335%|
|constant_no_between|98.074%|95.237%|
|top128_condition|97.562%|94.541%|

| Frozen contrast | Estimate | Joint95% interval | Lower > −.005 |
|---|---:|---|---|
|shared_minus_exact_learned|+0.000011|[-0.001711, +0.001732]|True|
|shared_minus_exact_random_mean|-0.001552|[-0.003274, +0.000169]|True|
|zero_minus_exact_learned|-0.000313|[-0.002034, +0.001409]|True|
|zero_minus_exact_random_mean|-0.002641|[-0.004362, -0.000919]|True|

Four noninferiority gates use0.5pp margins, joint95% maximum-absolute-deviation bands,4000 paired source/context bootstrap draws and eight-span resampling. Calibration uncertainty is conditional/fixed. Failed noninferiority is not proof of difference.

Descriptive normalized-output mean geometry: shared norm0.583543; receiving norm min/median/max0.534529/0.691064/1.538377; median cosine to shared0.793214; median relative shared-mean error0.622623. These raw mean errors need not predict full finite behavior.

Audits: exact mean transport relative error6.16e-15; full covariance decomposition2.9e-15; H450 replay absolute error1.09e-16; solve residual4.06e-15. All gates pass, arrays finite, no exclusions.

All four frozen noninferiority gates pass against constant-isotropic/exact-tail-mean. Shared-minus-exact differences are+0.0011pp learned and−0.1552pp randommean, with joint95% intervals[−0.1711,+0.1732] and[−0.3274,+0.0169]pp. Zero-minus-exact differences are−0.0313pp learned and−0.2641pp random, intervals[−0.2034,+0.1409] and[−0.4362,−0.0919]pp. Zero random has a detectable loss but remains inside the0.5pp tolerance. Noninferiority is to the constant-isotropic exact-mean variant, not directly to full exact covariance; margins cannot be chained into a stronger full-reference claim.

Shared-mean recovery is98.132% learned/95.444% random; zero-mean98.100%/95.335%; exact-mean98.131%/95.599%; full98.166%/95.744%. Omitting the whole between term gives98.074%/95.237%. Preserving head-versus-zero contrast therefore recovers some of the cost of dropping the term; the exact tail mean still adds modest signal. The result weakens the requirement for detailed receiving-specific tail means, not all context dependence. Exact top128 constraints, output-RMS geometry, tail mass and analytic response remain.

Step back: twenty live explanations and discriminating tests

| Hypothesis | Current standing / next discriminating evidence |
|---|---|
|1. Few likely-token constraints determine much behavioral compactness.|Supported within this scope; vary output entropy while fixing source response.|
|2. Generic tail penalties suffice at practical tolerance.|Strengthened by H449–451; direct full-reference confirmation remains.|
|3. Detailed tail anisotropy defines essential semantic directions.|Weakened at this tolerance; targeted rare-token tasks could differ.|
|4. Exact tail mean is essential.|Weakened; shared/zero pass conditional tolerance, with residual loss.|
|5. The head mean, rather than tail semantics, explains the between-term benefit.|Compatible; separately remove head versus tail means.|
|6. Output RMS transport is the dominant context dependence.|Plausible; freeze/swap RMS maps while retaining probabilities.|
|7. Probability-to-head-token identity matters beyond confidence.|Earlier controls support it; test within this simplified rule.|
|8. The64-token collapse reflects limited representational capacity.|Weakened by spectral diagnosis and baseline-rule rescue.|
|9. The64-token collapse reflects unstable extrapolation.|Supported, including unused receiving interventions.|
|10. The dense analytic source-response operation carries most explanatory information.|Unresolved key bottleneck; compress its input information and test transfer.|
|11. A small shared input subspace suffices for that operation.|Open; rank sweeps of common input projections are discriminating.|
|12. Low-variance context directions are essential to intervention responses.|Earlier evidence supports risk; compare activity-ranked and weight-ranked inputs.|
|13. Source and context require distinct coordinates.|Compatible with bilinear structure; compare role-specific projections.|
|14. Apparent source specificity reflects geometry rather than semantics.|Supported by synthetic controls; matched semantic interventions needed.|
|15. Native retained factors are a privileged stable computational core.|Partial support; control their contribution separately under simplified fitting.|
|16. Radial correction accounts for nearly all accuracy.|Weakened by component experiments; retain ablation controls.|
|17. A local predictor is accurate but cannot replace intact computation.|Still live; intervention accuracy is not intact substitution.|
|18. Rare contexts or source draws dominate pooled conclusions.|Observed repeatedly; preserve uncertainty and per-context tails.|
|19. Many bases implement equivalent output behavior.|Compatible; random spans work well but equality/semantic equivalence unproven.|
|20. These findings are specific to one late layer/head/model.|Unresolved limitation; depth/source/model transfer eventually required.|

Next prioritize10–12 rather than further tuning the output tail. Freeze the shared-mean/constant-isotropic output rule and test common input compression of the16 source-direction linear maps. Compare weight-ranked input directions with baseline-activity input directions on the remaining96 intervention-unused receiving contexts, using new sources and spans; include uncompressed maps and full output fitting as references. Calibration must use disjoint earlier baseline documents. Report parameter counts, source/receiving uncertainty and continuous fidelity alongside frozen gates. This directly asks whether the accurate response law is a compact computational unit rather than an expensive restatement of dense weights. Exact native/radial terms and baseline-forward costs must remain explicit. Broader semantic and model-wide goals remain open.

Scope/cost: shared mean requires1152 scalars plus one covariance-strength scalar; zero mean requires only the scalar beyond likely-token constraints. The prediction still uses receiving RMS geometry, full baseline probabilities to identify top tokens and mass, retained/radial terms and dense analytic response. No runtime benchmark or model-wide semantic explanation is claimed.
