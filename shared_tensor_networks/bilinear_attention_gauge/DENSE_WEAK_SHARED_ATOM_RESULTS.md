# Dense-coded weak rule: reduction fidelity and a shared-atom graph

## What was tested

Ten independently trained four-head bilinear-attention models predict four
count-sign labels from seven key tokens. Each of the 16 key tokens carries
*all four* raw bits, mixed into nonorthogonal numerical features. One query
returns A/B/C/D; a rarer query returns B/C/A/D. The rare A=0, B=1, C=0
combination (both D values) is absent from supervised training. Checkpoints
are selected by ordinary validation BCE. On fresh 30,000-example tests, the
native models have mean 99.9903% joint accuracy; the least competent seed
still reaches 97.35% joint accuracy on the rare held-out combination.

The trained weights define an exact `2 query × 16 key × 4 output` function.
An extraction check compares this function with the native forward pass in
float64. We multiply the D logit and bias by 0.1 *after training* to make a
competent, lower-energy output without changing its deterministic decisions.
This is a controlled scale intervention, not a claim that training naturally
produced a weak D output. The mean median D margin after calibration is 4.37.

We fit rank-three and rank-four functions under four reduction measures:
uniform weight-only SVD over the full token table; unlabeled empirical usage
from 100,000 independently sampled sentences; a 50:50 uniform/usage mixture;
and output-balanced weight-only SVD. The last measure normalizes named output
channels and therefore assumes those output coordinates matter separately.
Neither the reductions nor the later rotations use task labels. All four
measures see the exact weight-derived function, so this is *not* a data-only
recovery experiment.

## Reduction: global fidelity can discard a competent rule

Means over ten models at D scale 0.1, tested on independent fresh sentences:

| Rank-three fit | Agreement on D with native model | Joint prediction agreement | Global all-pair edit relative MSE | Worst-output all-pair edit relative MSE |
|---|---:|---:|---:|---:|
| Uniform weight-only | 64.47% | 64.47% | .00306 | .762 |
| Empirical usage | 64.47% | 64.47% | .00307 | .762 |
| Half uniform/usage | 64.47% | 64.47% | .00306 | .762 |
| Output-balanced weight-only | 83.79% | 66.29% | .09698 | .233 |

Rank three is a real resource tradeoff: output balancing retains more D
behavior but sacrifices behavior elsewhere. At rank four, all measures reach
at least 99.989% joint agreement and essentially complete D agreement with
the native model; mean global all-pair edit error is below .00006. Without
the post-training D calibration, a 1% global edit gate rejects rank three
in all seeds. With calibration, it accepts rank three in **10/10 seeds** for
each of the uniform, usage, and mixed measures, despite the D loss. A 1%
*worst-output* edit gate rejects rank three in 10/10 for every measure.

The same failure occurs when the global edit gate uses only pairs that keep
the raw D bit fixed: those rank-three fits pass in 10/10. This is an
intentionally stringent missing-counterfactual check, although those pairs
can still change the *mixed D feature* through A/B. A rank gate must specify
which outputs or behavior families it protects. Tiny aggregate error alone
does not certify them.

## Legibility: edit sparsity finds reusable atoms

At each faithful rank-four cut, we held the function fixed and compared
orthogonal coordinate rotations. SVD, key-dictionary L1, all-pair edit L1,
and edit L1 using only the 56 pairs that keep the raw D bit unchanged are
compared with 32 Haar-random rotations per cut. Each optimized rotation
uses eight starts and preserves its cut to numerical precision. The true
four raw-bit code and intended graph are used **only after fitting** to
score recovery.

The intended graph has eight query/output rows and four reusable raw-bit
atoms. A, B, and C rows each reuse two atoms; D rows reuse A, B, and D.
For each recovered dictionary we match four axes to the raw bits, then call
a row/atom edge present when its absolute coefficient exceeds 10% of that
row's maximum. Across 40 fixed cuts (ten seeds × four reductions):

| Exact basis | Mean matched raw-atom cosine | Graph cell accuracy | Exact graph-row fraction |
|---|---:|---:|---:|
| SVD | .789 | .719 | .000 |
| Key-dictionary L1 | .718 | .694 | .066 |
| All-pair edit L1 | **.997** | **.976** | **.903** |
| Same-raw-D-bit edit L1 | **.997** | **.968** | **.872** |
| Random rotation | .717 | .601 | .042 |

An oracle projection of the trained rank-four functions onto the four known
raw-bit directions leaves only `4.33e-5` mean relative residual and yields
100% intended graph-cell accuracy. Thus the trained *effective functions*
really do contain the overlapping structure; the sparse gauges recover
most of it without receiving that graph. The recovered all-pair D row has
relative atom magnitudes `[.428, .313, .050, 1]` for A/B/C/D versus
designed `[.45, .35, 0, 1]`. D therefore reuses A and B components; it is
not a standalone orthogonal semantic axis. Removing only the D-matched
rank-one component leaves D accuracy around 64.6%, while the other three
rules remain near 100%. This is a functional-factor deletion, not a physical
head or internal-node intervention.

The sparse bases also reproduce held-out direct raw-D-flip *D-output
responses* accurately: mean relative response error is `1.26e-5` for the
all-pair and same-raw-D-bit objectives. Crucially, SVD already gets about
`1.37e-5` on that narrow response test. The new gain is the recovered
**multi-atom membership graph**, not an improvement in that edit error.
The same-raw-D-bit objective withholds all eight direct D-flip edges, but
its fitting data and the weight-derived cut still contain D information;
it does not establish discovery from data with the D rule absent.

## Interpretation and next gate

This is evidence that an edit-defined legibility prior can identify a
reusable **function-level** dictionary in a trained attention model with
nonorthogonal token coding. It does not identify the physical attention
heads, prove a unique canonical form, or discover a wholly unseen task.
The graph threshold matters: at 5% and 20%, edit-sparse graph accuracy is
lower than at 10%. The optimizer is nonconvex and eight starts are not a
global-optimum certificate. The toy output labels and four-atom budget are
supplied by design; they are useful controls but stronger discovery needs
withheld task families and unknown group count.

The next compact test should use a shared dictionary with overlapping
groups and independent group gates, comparing it at matched functional
rank/error with these orthogonal gauges. It should hold out an entire rule
or edit family from *both* basis fitting and rank selection, then test its
prediction and causal response under a genuinely new composition. Stop
the toy line if the group model does not improve graph recovery or transfer
over the exact edit-sparse rotation; do not spend GPU time on scale alone.
Only after that gate should we try the analogous intervention-defined prior
on a selected 546M attention head, with disjoint corpus fitting and
validation and separate downstream KL checks.

## Reproduction

Run `train_dense_four_count_weak.py` for seeds 0–9, followed by
`dense_weak_rule_reduction.py`, `dense_weak_rule_gauges.py`, and
`analyze_dense_weak_rule_graph.py` in this directory. The ten checkpoints
and JSON training histories, full reduction and gauge JSON, cut and factor
NPZ files, and graph-analysis JSON are saved alongside these scripts.
The train script has a seed CLI option; the three analysis scripts process
all ten checkpoints. CPU runs use the `neuron_circuits/.venv` Python with
single-threaded BLAS.
