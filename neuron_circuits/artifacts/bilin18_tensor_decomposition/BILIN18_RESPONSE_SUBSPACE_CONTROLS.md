## H813 — Learned response structure beats random projections, but broad cross-task transfer is weak

Add matched rank-8 controls to H812 using the exact same internal lexical split, task-balanced SVD fitting, held-out items, doses, own/input-swap interventions, and causal replay. Cross-task means training only on the other task's training items. Random controls are Gaussian-QR orthonormal eight-dimensional subspaces with fixed seeds 8130,8131,8132, shared across tasks. No random-seed selection or behavior-based fitting is performed.

### Fidelity against the tested module contribution

The denominator below is RMS output difference between full-response and zero-module-response controls. It prevents the small role of L6 MLP on IOI from making a poor representation look faithful merely because much of the total source effect bypasses that module.

| Rank-8 representation | Copy input relative error | Copy behavioral relative error | IOI input relative error | IOI behavioral relative error |
|---|---:|---:|---:|---:|
| Shared, trained on both | .392 | .223 | .459 | .357 |
| Trained on same task | .373 | .222 | .433 | .335 |
| Trained only on other task | .915 | 1.016 | .943 | .978 |
| Random, three seeds | .993–.997 | .994–1.005 | .995–.997 | .976–1.068 |

Learned same-task and pooled subspaces carry useful structure beyond random projections. A rank-8 subspace from the other task, however, has overall behavioral error approximately as large as deleting the MLP response. This weakens the interpretation that the pooled subspace represents a generally interchangeable set of computational directions. It is compatible with a pooled dictionary containing different useful directions for different response regimes.

The cross-task basis reconstructs some more vector energy than random projection, but that modest geometric improvement does not translate into good total causal fidelity. Same active dimension does not match total storage: two same-task rank-8 dictionaries have sixteen columns versus eight for a shared dictionary. H812 separately reported storage-matched shared8 versus specific4. Three random seeds are controls, not a precise random-baseline distribution estimate.

### Partial asymmetric transfer remains

Cross-task transfer is not uniformly absent. On copying, full input-swap contrasts at .05/.10 are -.03400/-.08146. The IOI-trained basis produces only -.00073/-.00146, essentially losing the signature. Random contrasts are also near zero, while the pooled basis retains -.02970/-.06781.

On IOI, full contrasts at .25/.50 are +.00595/+.01966. A copying-trained basis retains +.00236/+.00877, larger than the three random controls (near zero), although its overall response fidelity remains poor. Thus some context-sensitive component may transfer asymmetrically. This is a narrow metric-specific finding on three held-out items, not proof of a reusable semantic atom. The pooled basis retains +.00380/+.01199, also imperfectly.

### Geometry

The rank-8 task subspaces have principal-angle cosines [.682,.310,.256,.159,.124,.106,.086,.037]. These quantify their best-aligned combinations of directions: there is some alignment, but most combinations have weak overlap. Geometric overlap neither identifies shared atoms uniquely nor proves causal reuse. The principal directions were not selected or intervened on in this experiment.

### Verification and research implication

1248 model forwards completed. Full-response historical replay is exact in the log-probability metric. Tracked hashes unchanged, lexical train/holdout sets disjoint, paired perturbation directions exactly matched, and basis orthogonality error at most 2.34e-15. All outputs finite. Top-1 accuracy remains 95.8% copy and 100% IOI across controls, underscoring its insensitivity to response fidelity here. Reserved test untouched; natural-document pool remains 26,112.

This is an internal fit holdout on previously examined panels, with only three held-out lexical items per task and two dose/arm variants pooled for RMS metrics. It validates useful learned low-dimensional structure only partially; it does not validate a manifold, shared dictionary atoms, taxonomy, or DAG.

Next test whether group selection can be learned without supplying task labels. Fit a two-group low-rank response model with a fixed total dictionary budget and compare it with pooled and task-oracle baselines under the same causal replay. Train gates only from response geometry, preserve lexical grouping, and report whether held-out selection aligns with task or other factors. If automatic groups outperform a flat subspace, then shared/private atom extensions have a concrete response benchmark; if they fail, do not infer meaningful manifolds from the presence of clusters alone.
