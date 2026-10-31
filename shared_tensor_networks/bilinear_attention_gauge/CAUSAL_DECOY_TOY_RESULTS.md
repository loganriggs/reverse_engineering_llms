# Exact two-head control for reduction measure and legibility prior

## Construction

`toy_causal_decoy_matrix.py` enumerates all 144 ordered pairs of a
12-token vocabulary. Each head is a scalar query-key score multiplied by
a key-dependent value vector. The first head has a common rule, a
high-amplitude rare rule, and frequent contexts where its two rule
coordinates coactivate. Its physical two-dimensional basis is rotated
by three different angles while its token function stays fixed. The
second head has a very large hidden response on one punctuation-like
pair, but its output decoder gain is only .005. Its output effect is
small. All rank-one fits and gauge scores are computed from the exact
pair table; there is no sample noise or trained model in this control.

## Reduction measure: which rule survives one axis?

The table averages three physical gauges. Rank one means one projected
latent coordinate, optimized over 721 head-0 angles plus the decoy-head
axis. Relative squared output error is evaluated on exact uniform and
declared usage pair measures. The uniform floor is the mixing weight in
`floor × uniform + (1-floor) × usage`.

| Fit measure | Uniform-pair output error | Usage-pair output error | Rare-rule error | Common-rule error |
|---|---:|---:|---:|---:|
| Uniform / weight-only | **.168** | .312 | **.073** | .927 |
| Usage | .358 | **.059** | .554 | **.446** |
| 10% uniform floor | .325 | .062 | .497 | .503 |
| 25% uniform floor | .276 | .079 | .405 | .595 |
| 50% uniform floor | .210 | .142 | .254 | .746 |

The one-axis bottleneck makes the tradeoff unavoidable. Uniform pair
weighting largely retains the rare rule; usage weighting favors the
frequent common/coactive behavior and loses much of the rare rule. The
floor does not grant free fidelity: it buys rare-rule coverage at a
cost on the usage measure. This is an exact, data-free definition of
weight-only importance over the declared vocabulary. The usage law is
invented for this toy, not estimated from natural language.

The reduction objective also matters. If the one-axis choice minimizes
**hidden-response** squared error, it selects the high-amplitude decoy
head for all five measures and all three gauges. That axis has essentially
unit relative **output** error because it omits the actual rule outputs.
An output-aware function metric does not select the decoy. This is the
controlled analogue of the real model's punctuation heads having crisp
input-edit responses but small downstream paths; the toy is not evidence
that those real heads implement the toy's mechanism.

## Exact-function gauge: what makes the axes legible?

Now retain both coordinates of the first head. Any orthogonal rotation
preserves its output function exactly. We scan 721 angles and optimize
four coordinate scores; the table gives mean alignment with the two
known rule axes across three physical gauges. Alignment 1 means a
permutation/sign of those axes; .707 is a 45-degree mixture.

| Gauge objective | Mean known-rule alignment | What it selects |
|---|---:|---|
| Native-basis coefficient L1 | .877 | Physical axes, dependent on initial gauge |
| Pair-response L1 | .707 | Coactivation sum/difference axes |
| Output-contribution L1 | .707 | Same mixture in this orthonormal readout |
| Designed edit-response L1 | **1.000** | Separate common and rare rule axes |

The response-sparsity failure is substantive: common contexts activate
both rules together, so their sum/difference coordinates are sparser
under the full pair distribution. The edit ensemble instead contains
isolated common and rare substitutions, which favor separate rule
coordinates. This does **not** show unsupervised task discovery: the
input edits were selected knowing the two rule types, although no output
labels enter the edit score. It also shows that downstream output L1
need not add information over hidden-response L1 when the decoder has
equal norm on the retained subspace.

The toy establishes algebraic and optimization controls for the larger
program, not a recommended universal prior. The next trained toy should
learn the heads and rules from language-like data, hold out entire rule
combinations and edit families, and ask whether a proposed prior predicts
behavior under new interventions at matched function error. A metric or
prior that simply recovers the engineered toy axes has not passed that
test.

Reproduce with `OPENBLAS_NUM_THREADS=1 python toy_causal_decoy_matrix.py`.
The complete per-gauge, per-measure scores are in
`toy_causal_decoy_matrix.json`.
