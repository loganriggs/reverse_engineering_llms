## H516 — Expanded-support Fisher weighting gives modest gains without collapsing the rank spectrum

Fit the full-path output Fisher metric in the1,023-dimensional supported basis of H514's pooled1,024-document source covariance, using the same64 prescribed training receivers as before. For each fixed rank36/72/144/288/576/1,023, construct the covariance-weighted oblique map from the leading eigenvectors of C^(1/2) F C^(1/2). Keep the original reconstruction mean and evaluate native full attention-write replacement on the reused H512 natural panel. Compare with H515 PCA at each identical rank. No rank or regularization parameter is selected after outcomes.

| Rank | PCA mean KL | Fisher mean KL | PCA recovery | Fisher recovery | Relative KL error reduction |
|---|---:|---:|---:|---:|---:|
|36|0.0037097|0.0035856|68.859%|69.901%|3.35%|
|72|0.0025864|0.0025376|78.288%|78.698%|1.89%|
|144|0.0016872|0.0016316|85.837%|86.304%|3.29%|
|288|0.0008769|0.0008187|92.639%|93.127%|6.63%|
|576|0.0002643|0.0002334|97.781%|98.041%|11.68%|
|1023|0.0000126|0.0000126|99.894%|99.894%|0.00%|

Fisher has lower finite KL at every truncated rank in this panel, with recovery gains of about0.26–1.04 percentage points. Because the residual errors shrink with rank, the relative KL-error reduction reaches11.68% at rank576, despite only a0.26-point recovery gain. These are exploratory point estimates on reused data, not fresh superiority or equivalence claims. The earlier fresh rank36 comparison under the smaller covariance support did not establish an incremental Fisher gain; this diagnostic does not replace that result with a confirmation claim.

The broad rank spectrum persists. Rank36 Fisher recovers69.90%, rank14486.30%, rank28893.13%, and rank57698.04%. On the prescribed grid, both PCA and Fisher first exceed95% recovery at576; that is not a confirmed rank choice or a lower bound on nonlinear representations. Full supported rank1,023 gives the same99.894% recovery for both methods, as required when both reconstruct the empirical source span.

The maps improve the training-local objective more than they improve this natural-input evaluation. For example, rank144 training recovery is93.74% for Fisher versus91.58% for PCA under the fitted calibration-local objective, while natural finite recovery is86.30% versus85.84%. These quantities use different distributions/approximations and denominators; their difference is not a direct generalization-error decomposition. Possible contributors include covariance estimation, metric estimation, source/receiver association, finite nonlinear effects, and shared reconstruction-mean choices.

A key assumption is now worth testing explicitly. The objective combines an average receiver Fisher metric with a source covariance, effectively treating source variation and receiver sensitivity separately. Actual attention sources and receiver contexts on ordinary text are correlated. Earlier donor experiments approximated independent pairing more closely. A mismatch in that pairing may limit how well the factorized objective predicts natural compression fidelity, even when the local derivative itself is accurate.

The oblique maps also amplify some source directions. Operator norms range from4.05 at rank36 down to2.03 at576, versus1 for orthogonal PCA; full supported rank returns norm1. The source covariance condition number is about5.69×10^7. Raw source retention is lower than PCA at truncated ranks. These facts do not invalidate the measured finite results, but they are relevant to distribution shift and estimator sensitivity. No ridge or outcome-based clipping was introduced.

Observed-token CE remains descriptive, with changes small relative to their paired sampling errors; no task-loss equivalence or improvement is established. All methods still retain the native MLP, earlier layers, and original attention activations for evaluation, so no implementation-speed or whole-model compression claim is made.

Audits: basis orthogonality error below4.56e-13; composed full-path derivative central-difference relative errors below1.12e-8; every Fisher training objective is no worse than PCA at matched rank; map idempotence passes1e-8. Full-rank source reconstruction differs from the empirical-span projection by relative5.92e-15 and full-rank KL replays PCA exactly. The full-rank operator-norm SVD used the library's convergence fallback and completed; no process was restarted on a timeout. All saved rows are finite and aggregates replay. No documents were added; total remains21,504.

Next H517: compare the same fixed source reconstruction errors under their natural source/receiver pairing versus prescribed random permutations of those errors across receivers. Use the frozen evaluation source-error pool, so each permutation preserves the exact marginal source-error vectors and receiver contexts. Compute local full-path Fisher errors for PCA and Fisher at representative predeclared ranks36/144/576 plus the mean-source baseline; use eight fixed permutations, not an outcome-selected permutation. Report natural versus permuted pooled local recovery and method gaps. This is an artificial-error-injection diagnostic, not a natural task-loss or semantic intervention claim. If method ordering/gains change substantially under permutation, source/receiver coupling is relevant; if not, investigate metric/sample estimation or alternative representation families. Use reused data without fresh confirmation claims or post-hoc rank optimization.
