

## H255 — Local behavioral compactness depends on entropy, not strongly on candidate truncation

H254 was concrete progress: equal-norm random/shuffled controls separated generic
readout sensitivity from additional native-error alignment. H255 tests whether
H253's compact oracle corrections mainly benefit concentrated predictions or are
an artifact of keeping only256 candidate tokens.

Before outcomes, fix supports64/256/1024 and ranks16/64. Each chart is selected
from the compiled state's predicted distribution and exact score gradients; all
charts are constructed before native residual computation. Corrections still use
oracle native error coefficients. Exact full-vocabulary KL evaluates every result.
Entropy bins are fixed in nats: low<2, medium[2,4), high>=4. Reuse128 H252 documents,
512 sites; bootstrap2000 whole documents retaining all sites, including masks.
No winning support or rank is selected. A prior reporting typo about left singular
vectors does not apply: code uses right singular vectors in residual coordinates.

| Compiled entropy | Sites | Local16: support64 | support256 | support1024 | Local64: support1024 |
|---|---:|---:|---:|---:|---:|
|All|512|77.63%|78.83%|79.28%|89.16%|
|Low (<2 nats)|172|92.09%|92.86%|93.09%|96.77%|
|Medium (2–4)|177|82.70%|83.77%|84.06%|92.47%|
|High (>=4)|163|57.78%|59.55%|60.41%|77.96%|

Values are ratio-of-sums KL error recovery relative to the compiled baseline
WITHIN each stratum, not uniform averages of per-site recovery ratios. All-support
rank16 range1.65 points; high-entropy range2.63 points. The locked <=5-point
support-stability gate passes overall and in all eligible bins. At support1024,
low-minus-high rank16 recovery is32.68 points, document-bootstrap95%[27.78,37.85],
passing the locked >=15-point entropy-dependence gate. This is an observational
stratum contrast, not a causal effect of entropy alone.

Top64/256/1024 retain mean86.50/92.90/96.95% probability mass overall. High-entropy
mass rises67.65→82.34→92.31%; low-entropy mass98.69→99.39→99.75%. Increasing support
therefore materially expands high-entropy coverage but barely closes its rank16
recovery gap. Exact full Fisher remains untested: even1024 candidates omit mass.
Wrong-context16 recovery at support1024 is9.05% low,12.85% medium,−2.61% high;
matched context still matters within each stratum. Negative recovery means the
correction increases KL. Prior H253 support256 numerical results reproduce.

**Interpretation.** Generic readout geometry plus concentrated predictions explains
a large part of apparent local compactness. Broader candidate support does not
remove the entropy dependence. This strengthens T13 objective-dependent relevance
and qualifies T09 conditional charts: strong pooled behavior does not imply equally
compact interfaces at uncertain sites. It does not show that internal MLP features
are merely tokens, nor that entropy causes the difference; error direction, size,
context and probability tails covary. Random-error controls from H254 provide
complementary evidence, not an additive explanation of these percentages.

**Next locked comparison to replicate.** Use a fresh128-document panel disjoint by
complete-document token hash from1024 discovery, original64 test and current128
test. Freeze support256 and ranks16/64 as the original H253 protocol, not the best
H255 variant. Repeat native, three isotropic and three shuffled equal-norm controls
at scales.1/1, matched/wrong charts, and fixed entropy bins. Test generic sufficiency,
additional native alignment and entropy dependence jointly. Document-level
bootstrap; same-corpus freshness only. A new population could weaken any current
account. Coefficient prediction and cost-efficient routing remain open.

Four directly relevant tests pass (stratified bootstrap arithmetic, projection,
score-gradient and reconstruction); full suite not rerun. Figure generated and
visually inspected. All H255 runs complete; broad research goal remains active.
