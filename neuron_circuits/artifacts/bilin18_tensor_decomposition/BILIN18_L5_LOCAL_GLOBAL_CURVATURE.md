## H767 — Per-context sensitivity predicts finite reconstruction KL much better than the global metric

Previous goal turn: progress. H766 completed and was recorded; its mixed results motivated a test that separates local approximation failure from failure of the averaged metric.

We froze 16 reused validation documents, two prefix lengths (32 and 120), and six reconstruction-error directions per context: adaptive write PCA and both Fisher geometries, at budgets 8 and 64. No models are fit. Scale each error from 1% to 100% and measure actual KL(native||edited). Estimate a local directional logit derivative using symmetric 1% perturbations; its variance under the full native vocabulary distribution gives the local Fisher quadratic prediction. Symmetric 2% differences check numerical stability. This local estimate uses no categorical sampling. Both local and global predictions refer to the same frozen direction. KL uses the same-run zero-edit reference to reduce batch-dependent numerical differences.

| Budget | Edit scale | Mean actual KL, nats | Local prediction RMS error / actual RMS | Global prediction RMS error / actual RMS |
|---|---:|---:|---:|---:|
| 8 | 0.1 | 8.12825e-06 | 0.40% | 99.03% |
| 8 | 1.0 | 0.000837553 | 7.09% | 97.84% |
| 64 | 0.1 | 2.63183e-07 | 0.73% | 97.61% |
| 64 | 1.0 | 2.66633e-05 | 3.21% | 97.46% |

The global metric fails to predict individual directional KL even at small steps. The per-context approximation stays much more accurate through full-sized reconstruction edits. This makes large-step nonlinearity an insufficient explanation for the global metric's poor performance. It favors investigating context-dependent sensitivity and its relationship to reconstruction errors before adding more group structure.

The conclusion is narrower than proving that context averaging alone is responsible: the global metric also has Monte Carlo error, training-context sampling error, and a different context distribution. Those remain confounded. The local diagnostic uses access to each evaluation context's model response; it is not itself a deployable featurizer or evidence of held-out learned improvement.

Robustness checks (complete values saved in companion JSON):
- Budget 8: local curvature estimates at the two finite-difference steps differ by 0.08% relative RMS. Median full-step relative prediction error is 2.5% local versus 68.0% global. Removing the three largest-KL cases leaves 38.1% versus 98.2% RMS error.
- Budget 64: local curvature estimates at the two finite-difference steps differ by 0.56% relative RMS. Median full-step relative prediction error is 1.4% local versus 60.5% global. Removing the three largest-KL cases leaves 4.5% versus 160.8% RMS error.

Large-KL cases influence the pooled RMS, especially at budget 8. The median comparison still favors local sensitivity, but some full-step errors are substantial. Directional-derivative vector agreement is less stable in small-signal cases; curvature agreement is the relevant additional check, not a blanket claim of exact derivatives. The 1% point is partly an internal consistency check; the independent 10–100% points provide the finite-step test. No significance claim is made from this small reused panel.

Audits: 1,728 model forwards, finite outputs, immutable inputs/model hashes, native replay below 1e-4. No reserved-test evaluation; unique natural document pool stays 26,112.

### Refreshed hypothesis set: 20 live alternatives

1. Context-specific downstream sensitivity determines which directions matter. Leading explanation; compare local matrices across contexts.
2. Reconstruction error correlates with sensitivity, defeating a single average metric. Leading explanation; compare matched versus shuffled context/error pairings.
3. Rare high-sensitivity contexts dominate a useful objective. Plausible; report document-level tails and equal-weight versus tail-weighted fitting.
4. Finite intervention nonlinearity dominates. Weakened on this panel, remains plausible for larger causal swaps; repeat scale tests there.
5. Global Fisher Monte Carlo noise dominates. Unresolved; compare independent sample halves along these directions or increase sample count on fixed contexts.
6. The 128 training contexts undersample relevant regimes. Plausible; increase contexts at fixed total sampling cost.
7. Prefix-position distribution shift matters. Plausible; compare length-matched metrics and pooled metrics.
8. Output-aware scalar features need direct context-conditioned training rather than metric PCA. Plausible next model only after diagnosing 1–7.
9. Useful units are mostly scalar, with little reusable block structure. Supported relative to tested fixed blocks; not a universal claim.
10. Genuine groups exist but energy-based gates select them incorrectly. Open; compare reconstruction-aware and output-aware gates on a fixed dictionary.
11. Hard selection creates unstable support changes between nearby endpoints. Open; measure endpoint support turnover versus causal error.
12. A good feature basis changes with computational regime. Plausible; infer regimes without task labels, then test transfer and sensitivity.
13. Hierarchical gates capture semantic implication without improving reconstruction. Open; test implication separately from additive contributions.
14. Additive refinements are more appropriate than nested subspaces. Open; synthetic recovery plus matched real-data controls.
15. Shared computations require nonlinear DAG nodes rather than shared directions. Open; held-out compositions and node-removal selectivity are decisive.
16. Dictionary non-identifiability explains seed instability. Plausible; compare subspaces and intervention equivalence, not atom alignment alone.
17. Upstream head interactions make isolated feature effects misleading. Supported by prior nonadditivity; test joint intervention predictions.
18. Task-relevant directions are low variance and omitted by reconstruction objectives. Plausible; test causal error versus omitted variance and local sensitivity.
19. Some apparent manifold structure reflects lexical/template nuisance. Open; cross-template and nuisance-matched tests remain required.
20. There is no compact universal set of units at this site; useful units are conditional and task-relative. Plausible fallback, not established; test cross-regime transfer and complexity curves.

Priority: first discriminate context/error coupling, metric sampling noise, and context coverage (1,2,5,6). A matched-versus-shuffled local metric diagnostic is especially informative, but needs reliable directional evaluations or full local matrices; do not substitute shuffled scalar curvatures for the required cross-context quadratic forms. Then train a context-aware or direct-output objective only if the diagnosis supports it. Preserve the downstream goal of verified manifolds, diverse task transfer and reusable computation; do not equate a better KL surrogate with that goal being solved.
