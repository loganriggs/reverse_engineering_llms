## H507 — Correcting the metric path helps, but leaves most joint error

Refit each head's rank4 source map using a local output Fisher metric that includes the direct residual contribution. For source direction B, the state derivative is B+J_MLP B before the final RMS normalization, vocabulary projection, softcap, and probability weighting. Use the same64 training receivers as H504, with the same calibration source covariance and rank. This changes the downstream metric path without changing training contexts or source rank. Maps are still fitted separately per head.

Evaluate joint reconstruction of all nine natural head sources on the same512 H499 contexts. Compare both MLP-input-only replacement and full pre-MLP replacement, with previous frozen maps replayed. The full-path map is fixed before evaluation. These are reused-panel diagnostics with no fresh confirmation, uncertainty claim, or best-method selection.

| Evaluation path | Source map | Mean KL | Recovery vs mean replacement | Observed-token ΔCE (paired SE) |
|---|---|---:|---:|---|
|MLP_only|PCA|0.001354|76.45%|-0.000375 (0.002445)|
|MLP_only|pooled|0.001269|77.94%|-0.000444 (0.002361)|
|MLP_only|full_metric|0.001284|77.68%|+0.000004 (0.002358)|
|full_pre_MLP|PCA|0.005837|50.62%|+0.005061 (0.004711)|
|full_pre_MLP|pooled|0.006208|47.49%|+0.002348 (0.005037)|
|full_pre_MLP|full_metric|0.005747|51.39%|+0.003690 (0.004945)|

On the full path, correcting the metric reduces KL from0.006208 for pooled MLP-metric maps to0.005747, a7.43% relative error reduction. This is only about1.56% lower than PCA's0.005837. Normalized recovery rises from47.49% to51.39%, compared with PCA50.62%. The improvement over the MLP-only metric supports path mismatch as one contributor, but correcting that mismatch does not produce high joint fidelity at this rank.

On MLP-only evaluation, the full-path map is slightly worse than the MLP-trained pooled map (KL0.001284 versus0.001269). This is consistent with objective-specific representation choices, but the small differences on reused data do not establish a reliable Pareto frontier. Observed-token CE changes remain small relative to their sampling errors; no task-loss improvement or loss-equivalence claim is made.

The remaining error has several live explanations: rank4 per head may be insufficient, the metric may still be estimated imprecisely, sources and receiver sensitivity may be correlated, and errors from different heads may reinforce one another. H506 already showed that joint KL exceeds the sum of individual replacement KLs. That need not require nonlinear interactions; local quadratic KL contains cross terms between head errors. Correcting each head's individual metric cannot directly optimize these joint terms.

Audits: for each head, a full-path logit directional derivative is checked against double-precision central differences at two step sizes. Every new map's source-covariance-weighted local full-path objective is no worse than the previous pooled MLP-map objective on the same metric. Previous H506 per-case conditions replay within prescribed tolerances; all new arrays are finite and aggregate metrics replay. Full-path matrices/operators/eigenspaces are archived before evaluation. No new documents were used; tracked total remains19,456.

This result narrows the current interpretation: defining a representation by its downstream behavior requires the correct computation path, but correct path coverage alone does not make independently compressed components compose well. It still does not identify unique axes or prove that a particular head partition is the right set of computational units.

Next H508: decompose the actual-pair local full-path output-Fisher error for joint natural replacement into the nine per-head diagonal terms and their cross terms. Compare mean replacement, PCA, pooled MLP-metric, and full-path-metric maps; retain every head pair. Verify that the sum of the Gram matrix matches the Fisher error from the summed source direction, and compare it with finite joint KL. This distinguishes correlated local errors from finite nonlinearity and indicates whether a joint cross-head representation is more informative than more per-head fitting. Use the same frozen maps and data and label results diagnostic. If cross terms dominate, a joint covariance-and-metric construction is the next candidate; if diagonal errors dominate, increasing or reallocating per-head rank is the more direct test.
