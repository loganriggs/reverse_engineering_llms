

## H253 — Context-specific downstream charts recover behavior with little residual energy

Hypothesis: global restoration fails partly because relevant output directions vary
with context. Select each local basis from the COMPILED final residual state,
before computing native errors. Take exact log-probability gradients (final RMS,
vocabulary head and softcap included) for the top256 compiled-probability tokens;
weight by sqrt(probability), take right singular vectors in residual coordinates,
and project the native-minus-compiled error orthogonally. All coefficients remain
ORACLE values. Selection uses neither native errors nor observed next-token labels.

The128 H252 documents/512 sites are reused adaptively, not a fresh replication.
Wrong-context controls cycle one document while preserving token position. Frozen
H251 global Fisher maps are tested both as originally fitted oblique maps and as
orthogonal projections onto their decoder spans. This separates basis selection
from projection convention. Per-state vocabulary evaluation and SVD are expensive.

| Oracle correction | KL error recovered | Native top1 agreement | CE increase | Residual energy recovered |
|---|---:|---:|---:|---:|
|Compiled only|0%|85.16%|.14066|0%|
|Local4|60.82%|96.48%|.06500|.71%|
|Local16|78.83%|99.61%|.02727|2.14%|
|Wrong-context16|7.05%|85.55%|.13868|1.46%|
|Global Fisher16|44.61%|87.89%|.08454|26.73%|
|Global Fisher span16, orthogonal|27.11%|85.16%|.10699|32.69%|
|Local64|88.73%|100%|.01567|6.10%|
|Wrong-context64|18.02%|85.94%|.12803|4.90%|
|Global Fisher64|64.47%|91.60%|.05778|46.41%|

The locked rank16 small-repair gate FAILS: KL recovery is below80% and CE increase
exceeds .02, despite >98% top1 agreement. Context-specificity and both-convention
global-advantage gates PASS. Local16-minus-wrong16 KL is−.07211, paired document
bootstrap95% [−.08365,−.06191]; local-minus-global Fisher16 is−.03439
[−.04077,−.02884]. Local64 satisfies those numerical repair thresholds descriptively;
the predeclared gate was rank16, so it remains failed. Rank64 CE interval
[.00266,.02974] also makes uncertainty material. Full oracle recovers native output
numerically. Three directly relevant tests pass (new projection test and two prior
score-gradient/reconstruction tests); previous32-test suite plus new test is33,
but the whole suite was not rerun in this step.

Top256 tokens retain mean92.90% probability mass, minimum48.77%, median96.87%.
The basis is a truncated Fisher construction, not full Fisher. Top4/16/64 retain
57.92/77.65/91.91% of this truncated spectrum on average. Maximum chart
orthogonality error2.03e-6. The config initially called these left singular vectors;
the implementation correctly uses RIGHT singular vectors in residual space.

**Interpretation and competing explanation.** T09 context-dependent directions and
T13 objective-dependent relevance gain support: a very small fraction of residual
energy can restore most distributional behavior. However, this may reflect generic
softmax/readout geometry rather than learned MLP computational primitives. Local16
restores99.12% of native top-two margin squared error, but only23.30% of full-vocabulary
logit squared error. Wrong16 restores24.94% of the latter, despite negligible KL
repair. Thus sensitivity to likely tokens, not uniform logit reconstruction, drives
the gain. Neither the local basis nor its oracle coefficients establish semantic
features, unique coordinates, cheap routing or a forward computation.

**Next discriminators, prioritized by information gained.**
1. Generic geometry versus structured residual: apply equal-norm isotropic and
   shuffled residual perturbations at the same compiled states; compare local
   projection repair with the original structured error, normalizing by each
   perturbation's exact uncorrected KL. A similar advantage would favor readout
   geometry over MLP-specific error organization.
2. Probability truncation versus conditional compactness: freeze ranks and test
   top64/256/1024 candidates, report probability mass and high-entropy strata.
   Do not choose a winning candidate count on the evaluation set.
3. Robustness: freeze the protocol before distinct-document replication. Current
   H253 gains were observed on a reused panel.
4. Only after these checks, predict coefficients from available activations and
   measure total forward cost. Oracle sufficiency is not coefficient predictability.

All H253 runs complete. Broad research goal remains active.
