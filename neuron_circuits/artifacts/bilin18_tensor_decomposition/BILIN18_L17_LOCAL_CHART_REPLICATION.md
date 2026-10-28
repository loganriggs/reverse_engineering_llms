

## H256 — Frozen local-chart findings replicate on new documents

H255 made progress: a fixed support sweep showed strong entropy dependence, weak
sensitivity to candidate count. H256 freezes original support256 (not H255's best
support), ranks16/64, scales.1/1, original three Gaussian/shuffle seeds, and entropy
bins<2,[2,4),>=4 nats before new outcomes. New128 complete documents exclude1216
unique hashes:1024 discovery,64 original test,128 H252 test. Each yields a fixed
random129-token window and four sites. Manifest exclusion audit passes. Text comes
from the same cached corpus; older studies may have used it, and this is not a
model-training holdout or cross-distribution test.

Reuse H254's runner with a separate output stem. Added site-level output permits
stratification without changing chart construction, errors, corrections or scoring.
All charts derive from compiled states before native errors are computed. Norm-
matched isotropic/shuffled controls, synthetic targets, exact full-vocabulary KL,
wrong-context charts and oracle coefficients retain their earlier definitions.
Three controls are pooled before recovery ratios; document bootstraps condition
on fixed random draws. No observed next-token labels select or score these charts.

| Scale | Error family | Local16 recovery | Wrong16 recovery | Local64 recovery |
|---|---|---:|---:|---:|
|.1|Native|83.00%|11.09%|91.13%|
|.1|Isotropic|70.56%|11.82%|83.85%|
|.1|Shuffled|74.87%|19.21%|86.63%|
|1|Native|81.21%|11.89%|90.17%|
|1|Isotropic|66.45%|11.40%|81.00%|
|1|Shuffled|69.09%|17.27%|82.73%|

**Replication gates.** Generic sufficiency PASSES, as in H254. Native-specialness
FAILS again: small-scale native-minus-shuffled gap8.13 points misses required10.
The separately predeclared directional-alignment gate PASSES: all four paired
native-control intervals exceed zero. At full scale native-minus-isotropic gap
14.75 points[11.92,17.36], native-minus-shuffled12.12[9.70,14.33]. At .1 scale gaps
12.44[9.37,14.91] and8.13[5.84,10.34]. These are pooled recovery contrasts, not
additive causal contributions. The original failed gate remains failed.

| Compiled entropy | Sites | Native local16 | Isotropic local16 | Shuffled local16 |
|---|---:|---:|---:|---:|
|Low (<2)|198|94.59%|83.98%|82.61%|
|Medium [2,4)|186|81.87%|70.94%|75.47%|
|High (>=4)|128|60.78%|47.33%|50.33%|

At full scale native low-minus-high gap33.81 points, document-bootstrap95%
[30.24,37.54], passing the fixed entropy-dependence gate. Entropy stratification
also matters for random/shuffled errors, strengthening the generic readout account.
The strata remain observational: other site properties covary with entropy.
H253's composite behavioral-repair gate also required CE/top1; those metrics are
not evaluated here, so >80% KL recovery is NOT a pass of that composite gate.

Full correction returns exact zero KL for every family/scale; maximum relative
site-norm mismatch1.88e-7. Five directly relevant tests pass (stratum bootstrap,
norm matching, chart projection, exact score-gradient and reconstruction).
Whole project suite was not rerun. All H256 runs complete; weights unchanged.

**Research synthesis.** The evidence now supports a robust conditional behavioral
interface, mostly driven by generic readout sensitivity and sharpened by native
error/context alignment. It does not establish a dictionary of internal MLP units:
charts are readout-derived, costly, and supplied oracle missing coefficients.
Neither compact output correction nor native alignment alone identifies semantics.

**Next discriminator.** Separate readout-metric concentration from internal
computation using a controlled temperature intervention: hold every residual state,
native/compiled error and vocabulary logit fixed; change only softmax temperature
for chart construction and evaluation, with matched and crossed temperatures.
If apparent compactness changes at the same internal computation, it is partly a
property of the behavioral metric. Report exact KL, candidate probability mass,
finite versus small-scale effects; this is a diagnostic altered readout, not an
improved native model. After establishing that boundary, return to forward-
computable coefficient prediction or causal source mediation, rather than declaring
readout directions to be MLP features. Broad research goal remains active.
