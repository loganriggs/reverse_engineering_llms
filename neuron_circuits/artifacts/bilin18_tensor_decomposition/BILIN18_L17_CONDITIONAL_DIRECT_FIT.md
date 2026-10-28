

## H296 — Direct partner optimization yields a small pooled gain and payload tradeoff

Freeze core96 and native cross coefficients; directly optimize a rank32 orthonormal partner projection in its complement. Four Adam runs cross weighted/random initialization and rates.003/.01,400 updates, batch128, checkpoints0/100/200/300/400, fixed minibatch seed. First768 discovery documents train, next256 validate; actual and angle full removals for all nine heads. Objective pools squared cross-response error normalized by pooled training energy. Thus larger-energy payload families receive more weight; this is not equal weighting of separately normalized family errors. Validation selects the best pooled cross fidelity.

Winner is weighted/rate.003/step200. Weighted/rate.01 selects step0, while both random-start runs select step400; not all trajectories have converged. No global optimum or impossibility claim. Training pooled fidelity rises14.67%→16.32%; validation15.86%→16.39%. Validation actual fidelity falls16.60%→14.08%, while angle rises15.31%→18.08%. This tradeoff is visible before diagnostic evaluation and must not be hidden by the pooled score.

| Run | Selected step | Pooled validation cross fidelity |
|---|---:|---:|
|weighted_lr0.003|200|16.39%|
|weighted_lr0.01|0|15.86%|
|random0_lr0.003|400|15.12%|
|random0_lr0.01|400|12.69%|

Evaluate frozen partners in float64 on the reused H293128-document diagnostic panel and original actual/angle seed. Re-orthogonalize exported float32 partners in float64. Proxy fidelity reproduces H295 within1e-6; evaluation overlap/orthogonality errors<1e-10. Existing partner-projection gradient/constraint test passes. An evaluator indentation error caused a module-startup NameError before evaluation began; its log is preserved, indentation fixed and the frozen evaluation completed. No training restart or outcome-based model reselection.

| Payload | Proxy cross fidelity | Optimized cross fidelity | Gain | Paired95% gain |
|---|---:|---:|---:|---:|
|actual|17.40%|14.89%|-2.51pp|[-2.87,-2.16]pp|
|angle|15.19%|17.40%|2.21pp|[1.33,2.95]pp|

Locked gates: direct_gain: FAIL, compact80: FAIL, retention: PASS. Direct gain requires10pp and positive lower95% in BOTH families; compactness80% in both; retention lower95% gain>=−3pp in both. No thresholds relaxed. Diagnostic bootstrap clusters all sites/heads by document. This is not a new holdout or intact behavioral test.

**Inference.** Direct empirical fitting does not rescue compact cross coverage under this budget. It makes a small pooled improvement by shifting emphasis across payload families. This weakens a simple claim that replacing the tensor proxy with pooled gradient optimization is sufficient. It does not prove broad dependence is irreducible: the objective weighting, nonconvexity, finite budget and unconverged random starts remain relevant. Dense compiled cross cost remains3,538,944 output coefficients at rank32, before reads/base. Native coefficients are still exact; no cheap learned surrogate or semantic unit has been identified.

**Next discriminator.** Test objective weighting at fixed rank and initializations before increasing capacity. Fit equal-weight mean of separately energy-normalized actual and angle losses, selecting by the worse validation family (with the original proxy available as a baseline). Preserve the pooled-loss run for comparison. If it removes the actual/angle tradeoff while improving both, pooling was material; if neither improves, reconsider partner parameterization or collect stronger optimization evidence before inferring distributed structure. Any candidate still needs a fresh panel and intact evaluation before promotion.

All jobs terminal. Frozen bases, curves, selection/payload audits and raw diagnostic errors retained under bilin18_l17_conditional_direct_fit* and _eval*. Original native model/compiler/base unchanged. Goal remains active.
