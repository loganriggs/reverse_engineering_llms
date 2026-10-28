## H506 — Joint source compression and the direct residual path

Replace all nine layer17 head-source contributions jointly, using the same frozen per-head rank4 maps from H505. Each centered source is reconstructed separately; sum the nine reconstruction errors e. The reconstructed pre-MLP state is p'=p−e. Compare two explicitly different computations: MLP-only replacement outputs p+MLP(p'), retaining the original residual bypass; full-pre-MLP replacement outputs p'+MLP(p'), replacing both paths. The latter replaces the attention write before the last MLP and residual addition. Earlier layers and the full native MLP remain unchanged.

The maps expose at most36 source coordinates across the nine heads, plus fixed calibration means. This is not a rank36 representation of the entire residual state or whole model. Source maps are still fitted separately per head using the MLP-path metric, not a joint full-path objective. Evaluate the same512 H499 ordinary-text contexts. No new fitting, selection, documents, or confirmation claims are introduced.

KL is original-output distribution to replacement-output distribution. Recovery is relative to replacing all nine centered sources by zero (retaining their calibration means), separately for each path. This denominator differs between paths, so absolute KL must accompany the normalized recovery. CE change uses the observed next token; positive means worse loss. All values are in nats except recovery.

| Path | Reconstruction | Mean KL | Recovery vs mean replacement | Mean ΔCE | Paired CE SE |
|---|---|---:|---:|---:|---:|
|MLP_only|mean|0.005751|0.00%|+0.002789|0.004590|
|MLP_only|PCA|0.001354|76.45%|-0.000375|0.002445|
|MLP_only|orthogonal|0.001493|74.04%|-0.000466|0.002550|
|MLP_only|original|0.001374|76.12%|-0.001393|0.002541|
|MLP_only|pooled|0.001269|77.94%|-0.000444|0.002361|
|full_pre_MLP|mean|0.011822|0.00%|+0.011244|0.006796|
|full_pre_MLP|PCA|0.005837|50.62%|+0.005061|0.004711|
|full_pre_MLP|orthogonal|0.006022|49.06%|+0.005067|0.004863|
|full_pre_MLP|original|0.006237|47.24%|+0.004922|0.004919|
|full_pre_MLP|pooled|0.006208|47.49%|+0.002348|0.005037|

For MLP-only joint replacement, PCA recovers76.45%, original Fisher76.12%, and pooled Fisher77.94%. The original Fisher map's individual-head advantages do not automatically compose into a joint advantage over PCA. Pooled Fisher has the smallest MLP-only KL among these fixed candidates, but the margin over PCA is small and this reused panel does not establish a general superiority claim.

Including the direct residual path increases absolute KL substantially. Full-path PCA has KL0.005837 and50.62% normalized recovery; original Fisher0.006237 and47.24%; pooled Fisher0.006208 and47.49%. Thus the current MLP-trained Fisher maps are worse than PCA on this full-path comparison. The mean-replacement KL also rises, from0.005751 to0.011822. We should not interpret the cross-path recovery difference alone as an additive fraction of information carried by the bypass.

This is an important scope limit: the metric used to construct the Fisher maps differentiated the MLP route while holding the residual bypass fixed. It did not penalize errors that matter through the direct residual route. A full-path metric must differentiate both: a source-direction derivative becomes (I+J_MLP) applied to that direction before the final probability readout. The result is consistent with metric-scope mismatch, but does not by itself prove that including the bypass in fitting will fix the error. Per-head fitting and source dependence may also limit joint reconstruction.

Joint KL is not the sum of the nine individual replacement KLs:

| Reconstruction | Joint MLP-only KL / sum of individual-head KLs |
|---|---:|
|mean|1.292|
|PCA|1.517|
|orthogonal|1.573|
|original|1.648|
|pooled|1.762|

The ratio is about1.52 for PCA and1.65–1.76 for the Fisher maps. This indicates that individual-head error summaries do not predict joint fidelity by simple addition. It does not isolate finite nonlinear interaction: even a local quadratic KL has cross terms when source errors combine. Local cross-term decomposition would distinguish that explanation from stronger nonlinear effects.

Observed-token CE changes remain noisy. The pooled full-path map has a smaller sample CE increase than PCA despite worse KL fidelity, but their separate standard errors do not constitute a paired comparison test. No CE improvement, equivalence, or reliable tradeoff frontier is established. All maps were constructed from probability-fidelity objectives, not CE optimization.

Audits and scope: saved H505 pre-MLP states, own head writes, labels, and frozen models were hash-checked; all10 path/method per-case arrays are finite and replay the displayed KL/CE aggregates. H505 already verified head-write summation and original receiver reconstruction. Mean replacement is the sum of all centered head contributions; the two paths differ only in whether the bypass uses p or p'. No new documents were consumed; tracked total remains19,456. This evaluation still computes the original source activations before reconstruction and retains the native MLP; deployment parameter/FLOP savings are not claimed.

Next H507: fit a full-path Fisher metric on the same64 prescribed training receivers used for the pooled MLP metric, with derivative I+J_MLP and unchanged source covariance/rank4. Refit all nine source maps, then evaluate the same joint natural replacement under both paths. Compare with the frozen pooled MLP-metric maps and PCA, without choosing heads or maps after outcomes. This directly tests whether the objective's missing residual route explains the failure. Treat the reused-panel comparison as diagnostic, and retain joint source-correlation/cross-term errors as a competing explanation. A future joint source metric could address those correlations if path correction alone is insufficient.
