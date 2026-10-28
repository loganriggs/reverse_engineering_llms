## H762 — Capacity helps causal fidelity, but reconstruction remains an incomplete objective

The H761 panel is reevaluated with frozen H754 write-PCA, adaptive write-PCA and three random orthogonal adaptive transforms at budgets8,16,32,64,128. No fitting or sparse-model retraining occurs. Only L5H5 is examined because H3's full effects were weak. All32 pairs, task/nuisance/identity edits, and query/source/both positions remain. Fixed PCA has global rankK; adaptive methods useK active coordinates per example from128 available axes.

| Final-query task cell | Full-effect RMS | Fixed PCA error, K8→K32→K64 | Adaptive PCA error, K8→K32→K64 | Random adaptive error, K8→K32→K64 |
|---|---:|---|---|---|
| Colon | 0.253 | 92.9%→52.6%→33.8% | 120.0%→47.2%→20.7% | 96.0%→66.2%→41.1% |
| Equals | 0.254 | 92.1%→39.7%→31.7% | 90.1%→60.9%→28.1% | 92.1%→88.0%→51.7% |
| Newline | 0.262 | 81.7%→57.6%→49.8% | 140.9%→123.0%→25.2% | 111.3%→83.8%→65.2% |

Errors are relative RMS changes in the intended-answer margin. Random values average three fixed seeds. The semicolon task effect remains below the predeclared0.25-logit ratio threshold: full RMS0.169, with K64 absolute errors0.086 fixed,0.082 adaptive,0.195 random. All source/both-position results remain in the artifact; no format or seed was selected for fitting.

Natural validation write-MSE at K64 is11.05% for fixed PCA,2.78% for adaptive PCA and7.35% for random adaptive. Those squared reconstruction errors are different metrics from the table's causal RMS errors and should not be compared as equivalent percentages. Nevertheless, strong natural reconstruction does not guarantee a faithful causal edit on the copying panel. Adaptive/data-derived geometry generally outperforms random geometry at high capacity, so the result is not entirely explained by retaining more arbitrary coordinates.

Improvement is not monotonic. Newline adaptive task absolute error rises from0.369 at K8 to0.419 at K16 before falling to0.322 at K32 and0.066 at K64. Similar reversals occur in some nuisance curves. More reconstructed activation energy can change downstream cancellations or add irrelevant directions; this experiment does not identify which mechanism causes each reversal.

Nuisance preservation also improves unevenly. At final-query newline, K64 absolute errors are0.090 fixed,0.058 adaptive,0.122 random against full RMS0.315. For equals, K64 errors are0.024 fixed,0.044 adaptive,0.117 random against full RMS0.104, so no relative error is promoted there. No single representation dominates every task/nuisance cell.

All methods at128 coordinates replay the full intervention exactly at the stored margin precision. This verifies that the pipeline can recover the full effect and supports omitted/altered coordinates as a source of the low-budget failures; it does not prove that all128 coordinates are independent computational units.

Interpretation: eight active coordinates are a restrictive budget for these upstream representations. Increasing capacity recovers much more effect, but appreciable error remains at64, and intermediate changes need not track natural reconstruction. Before introducing hierarchy, test whether task-blind downstream sensitivity identifies a better low-dimensional geometry than residual-write MSE. This connects to the data-based canonical-form idea: choose the equivalence criterion explicitly—activation writes, output distribution, or task behavior—and evaluate each separately.

Audits:9,984 forwards; H761 budget8 and full references replay exactly; every full-rank margin replays exactly; identity logit error<=2.17e-5; full donor-head vector error<=1.77e-5; all hashes unchanged and outputs finite. No original reserved-test evaluations; natural document pool26,112 unchanged. This is a capacity diagnostic on an already-inspected panel, not a new independent confirmation.

Next: estimate a task-blind local output-distribution sensitivity metric from natural training contexts, audit its gradients/numerics, and fit geometry using training activations only. Compare against the same frozen PCA/random baselines without selecting on the copying outcomes. Explicit sentiment/capital wording checks remain outstanding.
