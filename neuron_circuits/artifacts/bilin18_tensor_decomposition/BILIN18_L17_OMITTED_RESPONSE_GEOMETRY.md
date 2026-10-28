## H435 — Omitted directions versus cancellation

H434 independently confirmed an origin-dependent added-factor usefulness contrast, while union128 retained high raw fidelity but weaker finite behavioral coverage. H435 is a post-primary diagnosis on that same panel, not a new confirmation. Keep full response f, retained union response r, omitted response o=f-r, all factor groups and all32 baseline-selected contexts fixed. Evaluate raw squared norms and baseline Fisher energy across12 sources and all origin/destination combinations. No new documents or exclusions; total11968.

Write E(f)=E(r)+E(o)+cross(r,o), with the cross term including the factor2. Separately project o onto f in ordinary Euclidean coordinates for each source/origin: o=alpha*f+o_perpendicular. This separates an omitted magnitude component from a change of direction. Raw orthogonality does not imply output-Fisher orthogonality. Each decomposition is exact accounting, not unique causal mediation.

| Matched context stratum | Raw omitted/full energy | Fisher omitted/full energy | Omitted/full sensitivity per raw energy | Finite KL error/full effect | Retained–omitted cross/full Fisher energy |
|---|---:|---:|---:|---:|---:|
|Low RMS|4.09%|40.15%|9.81x|38.90%|+8.17%|
|Median RMS|2.96%|14.72%|4.98x|15.33%|+2.02%|

The low-RMS raw error fraction4.09% is amplified to40.15% under local probability-sensitive scoring: the omitted direction has9.81 times the full response’s Fisher sensitivity per unit raw energy. For median contexts that ratio is4.98. Finite-KL error fractions38.90% and15.33% are close to the local fractions40.15% and14.72%, respectively. Thus finite output curvature is not required to explain the large matched aggregate discrepancy here; this is not a claim of local/finite equivalence at every site or intervention size.

The pooled retained–omitted cross term is positive in every matched/crossed stratum. For matched low-RMS states, retained energy/full is51.68%, omitted40.15%, cross+8.17%. For median states these are83.26%,14.72%,+2.02%. Hence net destructive cancellation between union128 and its omitted response is not driving the aggregate error fraction in this panel. Individual examples and core-versus-added groups may still exhibit cancellation; this result does not overturn their earlier findings.

In low-RMS matched contexts, the omitted component perpendicular to the full response contributes80.84% of omitted Fisher energy, the parallel component3.45%, and their positive cross term15.71%. Median values are85.30%,9.88%,4.82%. These three accounting terms sum to100%; the cross term is not a separate feature, and perpendicular here is defined in raw Euclidean space. The large error is primarily directional rather than a missing scalar amount of the retained vector.

| Matched stratum | Original Fisher recovery | Best single retained-vector scale | Recovery with that scale | Per-example oracle scalar recovery |
|---|---:|---:|---:|---:|
|low|59.85%|1.0790|60.17%|68.29%|
|median|85.28%|1.0121|85.29%|88.87%|

These scalar bounds use full target responses to choose the best Fisher projection onto the retained-vector ray. They are deliberately oracle diagnostics, not deployable predictors or finite-KL validations. Even a separate best scalar for every example reaches only68.29% local recovery in the low-RMS matched stratum. Simple gain calibration therefore cannot supply the missing directional computation under this metric.

Numerical checks pass: full and omitted energies replay H434 within1.02e-15 relative error; both energy/cross decompositions close within7.2e-16. Independent pooled ratios retain H434 finite-KL endpoints. No previous scientific gate is changed.

Updated interpretation: a compact native group can capture nearly all raw activity while omitting small directions that matter disproportionately to likely token predictions. This limits an ontology built from high-variance factors alone. It does not establish that the missing directions are semantically coherent, universally shared, or low-rank enough for an efficient correction.

Next high-information test: use an input-derived local predictor of omitted response directions, frozen before evaluating their finite target errors, and test its transfer across the independent baseline strata. Compare a small shared output basis with baseline-conditioned directions and matched-size controls; account for all reads, coefficient computations and basis storage. A predictor that requires the full omitted response or a target-fitted basis is only an oracle ceiling, not progress toward a compact transferable computational unit. First measure whether the analytic local prediction is accurate specifically for the omitted residual; full-response accuracy alone need not imply that.
