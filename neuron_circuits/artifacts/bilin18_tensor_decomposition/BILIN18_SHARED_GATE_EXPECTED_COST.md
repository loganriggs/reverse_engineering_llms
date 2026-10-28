## H695 — Consequence-weighted gate training improves the objective, not exact recovery

Replace straight-through gate gradients and binary-label distillation with an exact differentiable expected cost. A detector supplies four independent Bernoulli probabilities. Sum the probability of each of16gate patterns times its precomputed reconstruction-plus-group-count cost. Train on training tables only, at original group penalty0.01, using the same architectures, seeds,3000steps and batch streams. Select by hard-threshold validation objective; held combinations are evaluated afterward. No teacher labels enter training or selection.

| Seed | Detector | Validation objective | Validation group recovery | Held objective | Held group recovery | Held relative MSE |
|---|---|---:|---:|---:|---:|---:|
|21|linear|0.04000|0.00%|0.04000|19.09%|0.00000|
|21|energy|0.02492|43.16%|0.03995|39.75%|0.00570|
|22|linear|0.04000|0.00%|0.04002|19.09%|0.00002|
|22|energy|0.02491|44.87%|0.04213|37.50%|0.00736|
|23|linear|0.04000|0.00%|0.04000|19.09%|0.00000|
|23|energy|0.02406|45.41%|0.04238|41.46%|0.00760|

Energy gates reach validation objective0.02406–0.02492, improving over straight-through training0.03545–0.03695 and search-label BCE0.04954–0.05127 in all three seeds. Exact search remains better at0.01600. Validation group recovery43.16–45.41% lies between straight-through1.42–11.38% and BCE86.82–87.01%. Held recovery37.50–41.46% remains inadequate, with held relative MSE0.57–0.76%. Linear gates again mostly keep everything open.

Expected stochastic cost is not the hard-threshold evaluation cost; both are recorded. Conditional independent probabilities do not restrict deterministic patterns in the limit, but can still affect optimization. Objective improvements do not establish a generally superior featurizer, uniquely identified atoms or manifold semantics. All methods use the same frozen representation learned with supplied gates. Only the gate fitting is label-free, and exact pattern enumeration scales exponentially in group count.

Implementation audits: pattern probabilities sum to1; double-precision finite-difference gradcheck passes; independently derived conditional-cost gradients agree within1.39e-17. All source hashes unchanged, finite results, no original-model forwards/new documents, reserved LLM test split untouched.

### Refreshed hypotheses after the alternative-featurizer pilots

1. Missing coactivation constraints explain some composition failures — strongly supported by H688/H689.
2. Excessive learned overlap creates rank deficits — demonstrated for two failed fits in H690.
3. Straight-through discrete membership updates can destabilize good fits — observed in H690; exact source of instability not isolated.
4. Local discrete repair can recover functional structure — supported on this teacher by H691, not yet broad reliability.
5. Supplied group gates conceal a substantial discovery problem — supported by H692–H695.
6. Coordinates can carry sparsity while gates become redundant — all-open controls strongly support this here.
7. Linear gates are poorly matched to sign-symmetric presence — consistent with repeated failures; not a general theorem from these runs.
8. Sign-symmetric detectors contain useful structure — supported by label-distillation recovery, but not complete inference.
9. Straight-through training misses good decisions available to the same family — supported by H694's large accuracy gain, with changed objective.
10. Classification and reconstruction objectives favor different mistakes — supported by H694/H695's tradeoff.
11. A positive group penalty deliberately drops weak true features — supported by exact inference and penalty sweeps.
12. Larger combinations make groups cheaper to omit because other groups cover shared atoms — plausible explanation, marginal-effect audit still needed.
13. Exact inference can provide useful training signals without teacher labels — supported conditionally, not end-to-end discovery.
14. Quadratic detector capacity limits exact gate inference — possible; optimization remains a confound.
15. Noise and coefficient dynamic range determine recoverable group identity — untested in these noiseless pilots.
16. Reconstruction-equivalent dictionary rotations prevent unique atom recovery — imperfect atom alignment with exact reconstruction supports nonuniqueness, not a full characterization.
17. Hierarchical implications improve nested-data recovery beyond flat overlap — trained benchmark still required.
18. Additive refinements and taxonomy gates make different testable predictions — implementation semantics checked; empirical comparison pending.
19. Shared computational nodes generalize and have selective downstream effects — supported only in the fixed-topology synthetic DAG; discovered topology untested.
20. Real LLM blocks may reflect lexical/contextual mixtures rather than clean concept manifolds — earlier atlas and interventions support caution; richer featurizers have not resolved it.

The most actionable supported explanations are1,2,5,6,10and11. Their implication is to keep structural recovery, task output faithfulness, reconstruction and sparsity as separate measurements. The next broadening experiment is a trained nested-structure benchmark comparing taxonomy union gates and additive refinements with flat controls, using known topology to validate semantics before attempting topology discovery. Then test noisy/shared-computation cases and joint discovery. The original early-layer manifold stability, geometry and selective causal investigation remains the destination; no alternative is certified ready as a superior LLM featurizer yet.

Broad goal remains active.
