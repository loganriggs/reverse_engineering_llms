

## H273 — Directly optimized partner-rank sweep and fresh-document transfer

H272 showed direct selection helps at rank32. H273 changes partner rank16/32/64 while retaining natural core32, exact native-minus-compiled coefficients, training population, minibatch schedule and optimizer grid. Rank32 reuses the frozen H272 model. Ranks16/64 each run the same four initialization/learning-rate combinations,400 updates, checkpoints0/100/200/300/400, first768 discovery documents for training and next256 for checkpoint selection. Rank changes computational cost; equal updates are not equal FLOPs. No convergence or global-optimality claim.

Before evaluating new outcomes, freeze a128-document panel excluding all1600 unique complete-document hashes in the six recent panels. Verify128 unique hashes, zero overlap and output file SHA256. These windows can occur in older studies or model training; no global freshness claim. Four sites per document and nine source heads. Evaluate actual removals and same-head perturbations matched for norm and angle to the state, scales0.1/1. Training uses actual full removals only. All selected models are frozen before fresh evaluation.

| Rank | Selected run / step | Optimized train | Optimized validation | Fresh full actual cross | Fresh full actual total | Fresh full angle cross |
|---|---|---:|---:|---:|---:|---:|
|16|random0_lr0.003 / 300|47.99%|48.57%|44.86%|36.69%|21.00%|
|32|weighted_lr0.003 / 300|54.43%|54.45%|50.79%|39.09%|26.60%|
|64|weighted_lr0.003 / 300|62.33%|61.04%|58.00%|41.68%|33.93%|

Weighted-surrogate baselines and optimized small-edit results:

| Rank | Fresh full actual weighted cross | Fresh full angle weighted cross | Fresh small actual optimized cross | Fresh small angle optimized cross |
|---|---:|---:|---:|---:|
|16|35.20%|23.74%|46.10%|20.27%|
|32|43.26%|28.95%|51.91%|26.03%|
|64|50.98%|35.86%|59.02%|33.46%|

Locked gates: capacity_gain: FAIL, compact80: FAIL, replicated_gain32: PASS, angle_transfer: FAIL.

Capacity gate requires optimized64−32>=10 points with positive paired document CI at BOTH actual scales. Compactness requires80% cross fidelity at64 both actual scales. Replication gate requires optimized32−weighted32>=5 points and positive paired CI at both actual scales. Angle transfer requires optimized64>=50% at both angle scales. No thresholds adjusted.

- actual_s0.1, capacity64_cross: 7.11 points [6.55, 7.69], paired document-bootstrap95%.
- actual_s0.1, capacity32_cross: 5.81 points [5.09, 6.33], paired document-bootstrap95%.
- actual_s0.1, gain32_cross: 7.49 points [6.67, 8.29], paired document-bootstrap95%.
- actual_s1.0, capacity64_cross: 7.20 points [6.65, 7.76], paired document-bootstrap95%.
- actual_s1.0, capacity32_cross: 5.93 points [5.27, 6.42], paired document-bootstrap95%.
- actual_s1.0, gain32_cross: 7.53 points [6.82, 8.25], paired document-bootstrap95%.
- angle_s1.0, capacity64_cross: 7.34 points [7.05, 7.61], paired document-bootstrap95%.
- angle_s1.0, capacity32_cross: 5.59 points [5.00, 6.22], paired document-bootstrap95%.
- angle_s1.0, gain32_cross: -2.36 points [-3.04, -1.70], paired document-bootstrap95%.

**Result that changes the next step.** Full actual optimized cross fidelity rises44.86→50.79→58.00% across ranks16/32/64. The64−32 gain7.20 points[6.65,7.76] is positive but below the10-point capacity threshold;80% compactness also fails. Optimized32−weighted32 gain7.53[6.82,8.25] passes the fresh-document replication gate. Full total residual fidelity rises36.69→39.09→41.68%.

Matched-angle behavior differs: optimized ranks16/32/64 give21.00/26.60/33.93% versus weighted23.74/28.95/35.86%. Thus actual-response optimization reduces angle coverage at all tested ranks. Rank64 fails the50% angle gate. This is evidence of objective/distribution specialization, not merely insufficient document generalization. It motivates a fixed-rank actual/angle training-mixture test before another rank increase.

**Interpretation.** Separate three questions: whether optimizing the basis transfers to new documents, whether additional partner capacity improves the same response family, and whether the resulting span covers altered payload directions. Positive capacity effects under a fixed protocol do not identify intrinsic dimension; incomplete coverage can still reflect optimization, core choice or the restricted projection family. Document transfer does not imply perturbation-family transfer. Matching norm and state angle removes two simple geometric differences but does not match all payload statistics.

These models still use exact coefficients and retain full mean-linear response. Cross-term fidelity uses cross-term energy; total fidelity includes within-core/mean-linear terms and omitted outside-core error. The rank sweep is not a semantic labeling experiment, a downstream repair, or a fully optimized forward implementation. The shared representation is one candidate unit of computation, not a uniquely identified ontology.

**Next information-gaining test.** Freeze rank32 and cross actual-only, angle-only and equally weighted actual/angle training losses with the same discovery split, initializations and update budget. Normalize each family by its training cross-response energy so the mixture cannot be dominated solely by larger targets. Use validation mixture loss for the mixture, preserving family-specific validation for single-family controls; report both performance axes rather than just a mixture score. This tests whether the observed tradeoff can be reduced at fixed capacity. Keep the H273 panel marked reused; use fresh random control realizations and reserve new documents for confirmation of any selected mixture. Compile and cost a stable frozen program and test downstream behavior after the transfer issue is understood. Preserve actual-only baselines, independent sampling and all failed gates. Do not infer semantic units merely from successful response compression.

**Audits.** Three targeted tests pass (QR gradient, partner-complement identity, quadratic partition and coordinate invariance). Rank32 was not retrained or reselected. All models/configs/checkpoint curves and raw document arrays retained. The rank-generalized runner defaults to32 for the original config. New-document manifest hash and exclusions verified. Full project suite not rerun. All jobs terminal, figure inspected, native weights and original compiler unchanged. Broad research goal remains active.
