## H508 — Individual-head reconstruction error dominates the joint gap

For each of512 natural evaluation contexts, compute the full-path probability-score tangent caused by each head's source reconstruction error. The full-path state derivative includes both direct residual and MLP contributions. Form a9×9 Gram matrix G with Gij equal to one-half the probability-weighted inner product of the two centered-logit tangents. Its diagonal is the sum of individual local errors; its off-diagonal terms capture reinforcement or cancellation. Sum every entry to obtain the local joint KL approximation.

Use the four prescribed maps: mean replacement, PCA, pooled MLP-metric, and pooled full-path-metric. All maps, contexts, and errors are frozen from H507. Compare the local sum with the previously measured finite full-path KL. This is a diagnostic on reused data, not a new causal intervention experiment or confirmation.

| Map | Local diagonal error | Net cross terms | Local joint error | Net cross fraction | Actual finite KL | Finite/local |
|---|---:|---:|---:|---:|---:|---:|
|mean|0.010404|0.001285|0.011689|10.99%|0.011822|1.0114|
|PCA|0.004918|0.001007|0.005925|17.00%|0.005837|0.9853|
|pooled|0.005256|0.001085|0.006341|17.11%|0.006208|0.9790|
|full_metric|0.004878|0.000972|0.005849|16.61%|0.005747|0.9825|

For full-path-metric maps, diagonal terms account for83.39% of local joint error and net cross terms16.61%. PCA and pooled MLP maps have similar net cross fractions, about17%. Thus cross-head reinforcement matters, but the larger component is individual reconstruction error. Fixing only net cross terms while leaving diagonal errors unchanged would not remove most of this measured gap. That is an arithmetic diagnostic, not an achievable intervention bound: changing a representation can change both terms.

Positive and negative cross terms coexist. For the full-path map, positive off-diagonal terms sum to0.001345 and negative terms to−0.000373, yielding net0.000972. The Gram matrices contain all pairs; no selected head pair is promoted as the complete explanation. These terms reflect overlap of output-score perturbations, not necessarily direct causal interactions between heads.

The local joint estimate closely matches finite KL: finite/local ratios range0.9790–1.0114 across the four conditions. For the compressed maps, finite KL is about1.5–2.1% below the local estimate. Strong finite nonlinear effects are therefore not needed to explain most of the aggregate joint error on this panel. Individual contexts can still differ; these are pooled comparisons, not pointwise error guarantees.

Audits: tangent of the summed source error matches the sum of the nine tangents within relative tolerance1e-4; summing Gram entries matches the independently computed direct tangent energy within1e-4. Per-context diagonals are nonnegative and all saved arrays are finite. Aggregate matrix sums replay the displayed local errors, and finite KL is read from hash-checked H507 per-case outputs. All512 per-context Gram matrices are archived for every method. No additional documents were used; tracked total remains19,456.

The evidence prioritizes rank allocation and per-head reconstruction before treating cross-head interaction as the dominant failure. Exactly four coordinates per head was a convenient fixed budget, not an established property of the computation. Heads differ substantially in source geometry and behavioral importance.

Next H509: keep the total coordinate budget at36, but allocate it across heads using only the frozen training covariance and full-path Fisher spectra. Rank the eigenvalues of C_h^(1/2) F_h C_h^(1/2) across all nine heads; select the largest36 globally and reconstruct each head using its selected prefix of eigenvectors. This minimizes the summed per-head calibration-local error at that budget, ignoring cross-head terms. Freeze the allocation before evaluation. Compare with uniform4-per-head full-path maps and PCA on the same natural joint-replacement panel, reporting absolute KL and observed-token CE. No test-outcome rank selection is allowed. This tests heterogeneous rank needs at fixed total coordinate count; it remains diagnostic on reused data and does not yet establish full-model parameter or compute savings.
