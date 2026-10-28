## H838 — Downstream fidelity preserves part of the scalar advantage, but local rankings are not reliable

Execute the H837-locked32-document panel without changes: both early MLP sites, probe0 (new random direction), probe8 (native attention-write direction), signed doses +/-0.1 times recipient pre-MLP RMS, all14 frozen representation candidates, plus natural finite, own analytic and zero-response controls. No documents, seeds or doses selected after outcomes.

### What the intervention measures

Add the scaled probe to the final-position attention output before the tested MLP. The natural finite reference runs that MLP normally. Approximation arms retain the same skip perturbation but replace only the MLP output by its recipient-native output plus the predicted local derivative response. “Zero” removes only this module response, not the original intervention or the model's native MLP output. Own analytic response separates finite-dose approximation error from representation error.

Measure KL(natural finite intervention distribution || candidate distribution) over the full vocabulary. Ratios below divide each aggregate error by the aggregate zero-response error. They are not percentages of model computation explained. Native states are recaptured at the exact prefix lengths; all representations use frozen H837 predicted states.

### Full-distribution results

Mean KL/zero-response KL, averaging the two signed doses and both seeds within each trained architecture. Seeds are not treated as independent statistical replication.

| Site / probe | Scalar48 | Scalar64 | Block8 | Fixed shared | Learned shared | Routed weighted PCA64 |
|---|---:|---:|---:|---:|---:|---:|
| block1 / random |.314|.441|.453|.524|.689|.420|
| block1 / attention write |.478|.554|.643|1.043|.995|.845|
| block2 / random |.524|.536|.537|.612|.659|.492|
| block2 / attention write |.301|.298|.366|.696|.634|.391|

Zero-response mean KL is5.25e-7,2.67e-7,1.80e-7 and1.87e-7 in these four rows. Absolute changes are small at the locked doses, but representations leave substantial residual error relative to the tested module response. Fixed shared on block1 attention-write is worse than zero-response on average, and learned shared is roughly tied. A wrong approximate response can be no better than omitting the tested response.

Scalar models generally outperform shared groups, but the clear local scalar64 ranking does not survive as a universal downstream ordering. Scalar48 has lower mean KL in three rows; the two scalar budgets are nearly tied in the fourth. Routed PCA and blocks can be competitive, particularly on block2 random probes. Individual seeds also reorder: for example block2 random block8 ratios are .633/.441 versus scalar64 .435/.638. Do not select one seed now and call it a generally superior representation.

### Ranking changes are not just a different local evaluation subset

Recompute local response error on these SAME32 documents and SAME probes. Scalar64 is locally better than scalar48 in every cell (e.g. .512 vs .534 on block1 random, .534 vs .560 on block2 attention-write), yet does not consistently have better final KL. Thus improved local Euclidean response prediction is an incomplete surrogate for downstream distributional fidelity. Context-dependent downstream sensitivity and output filtering documented earlier provide plausible mechanisms, but this batch does not uniquely attribute the ranking changes to one of them.

Scalar48 beats zero-response error on25–32 of32 document averages across the four cells; scalar64 on25–31. Some failures remain and are not excluded. Shared models beat zero on fewer documents in several cells, despite reasonable aggregate reconstruction. Full per-document/sign/seed results and logits are retained.

### Uncertainty

Paired4096-resample document bootstrap averages signs and seeds before resampling. All scalar48-minus-scalar64 intervals include zero. All scalar64-minus-routed-weighted-PCA intervals also include zero. Scalar48-minus-block8 excludes zero only for block1 random; the other three include zero. Scalar48-minus-fixed-shared excludes zero for block1 random, block1 attention-write and block2 attention-write, but not block2 random. These are descriptive32-document intervals without multiplicity correction. They support retaining several candidates rather than declaring a universal downstream winner.

### Verification

4448 whole-model forwards:32 native endpoint runs,4352 signed interventions (17 arms x256 site/document/probe/sign combinations), and64 zero-amplitude controls. Native pre-MLP replay relative error1.65e-6; normalized perturbed-input replay1.06e-6; zero-amplitude logit replay max2.01e-5, all within predefined1e-4 tolerances. Own analytic response KL is only1.84e-5–9.54e-5 times zero-response KL across cells, so the finite-dose approximation does not explain the representation errors. All input/model hashes unchanged, all metrics finite; tiny negative KL minimum-8.89e-16 is floating-point roundoff, not a negative divergence claim. Original reserved test untouched, no new corpus documents, natural-document pool26,112. Full outputs are saved.

### Research consequence

We now have implementation checks, training/coverage diagnostics, new-document local confirmation and a prespecified downstream test for exact-site scalar/block/shared alternatives. Scalar models are the strongest current local predictors and useful downstream baselines. Shared variants have no demonstrated advantage here, but their optimization and imposed grouping limits remain explicit. Block representations remain scientifically relevant if their groups are more stable or explanatory despite somewhat worse fidelity.

Return to the user's computational-unit question rather than continuing reconstruction tuning. Compare cross-seed stability of feature directions, block subspaces and activation supports for the frozen models; then test whether stable components predict selective causal effects on diverse tasks the model can perform. Keep scalar48/64, block8 and routed PCA as competitors where downstream intervals overlap. Refresh competing hypotheses, including downstream null directions, token-specific baselines, context-conditioned operators, unstable dictionary factorizations and genuine shared mechanisms. A useful feature ontology must survive these checks and held-out combinations; neither local reconstruction nor KL alone names the units of computation.
