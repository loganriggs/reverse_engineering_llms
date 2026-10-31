# A larger natural-text KL check of the frozen later-head reductions

The [bracket](BRACKET_REDUCTION_GROUP_FACTORIAL_RESULTS.md) and
[agreement](AGREEMENT_REDUCTION_GROUP_FACTORIAL_RESULTS.md) factorials
showed that the natural-corpus rank-four PCA cuts lose 80% and 72% of
their respective native finite head paths. That is an intervention
fidelity result, not a general comparison of language-model quality.
The earlier [rank pilot](LATER_HEAD_REDUCTION_METRIC_RESULTS.md) used
only 24 natural-text windows per corpus. Here we test whether the
corpus cut gains full-vocabulary KL fidelity on a larger, independent
panel.

We **freeze** the original five rank-four cuts for each selected head:
weight-only output-projection SVD, uncentered FineWeb head-state PCA,
task-edit PCA, equal-trace mixed moment, and random. The corpus basis
was fitted on just 256 length-32 FineWeb contexts; this experiment
increases **evaluation**, not fitting, data. Every method sees the
same held-out token windows. For each head, we evaluate 128 length-32
and 64 length-64 windows from each of disjoint-document FineWeb,
WikiText-103, and TinyStories pools: **576 windows per head**. Only the
final-position vector of one selected head is cut. The reference is the
native model's full-vocabulary next-token distribution, and the
reported KL is `KL(native || cut)`.

| Head / task | Measure | Equal-panel mean KL ↓ | Native top-1 flips / 576 |
|---|---|---:|---:|
| L13 H8 / bracket | Weight only | **0.000977** | **7** |
| | Corpus PCA | 0.002467 | 10 |
| | Task-edit PCA | 0.007031 | 17 |
| | Mixed | 0.002487 | 10 |
| | Random | 0.006590 | 16 |
| L11 H3 / agreement | Weight only | 0.001839 | **4** |
| | Corpus PCA | **0.001763** | 8 |
| | Task-edit PCA | 0.002375 | 14 |
| | Mixed | 0.001821 | 10 |
| | Random | 0.002813 | 11 |

Equal-panel mean gives each corpus × context-length cell equal weight,
irrespective of its 128 or 64 sampled windows. We also compare corpus
minus weight **within each identical context** and bootstrap those paired
differences 10,000 times. Positive differences favor the weight cut.

| Head | Length | Corpus | Mean ΔKL, corpus − weight | Paired 95% interval |
|---|---:|---|---:|---:|
| Bracket | 32 | FineWeb | +0.000381 | [−0.000137, +0.001189] |
| Bracket | 32 | WikiText | +0.000065 | [−0.000007, +0.000161] |
| Bracket | 32 | TinyStories | +0.002553 | [+0.000265, +0.005547] |
| Bracket | 64 | FineWeb | −0.000042 | [−0.000092, +0.000001] |
| Bracket | 64 | WikiText | +0.001294 | [−0.000076, +0.003955] |
| Bracket | 64 | TinyStories | +0.004687 | [+0.000171, +0.010860] |
| Agreement | 32 | FineWeb | +0.000275 | [+0.000033, +0.000631] |
| Agreement | 32 | WikiText | +0.000265 | [−0.000100, +0.000877] |
| Agreement | 32 | TinyStories | +0.000065 | [−0.000113, +0.000254] |
| Agreement | 64 | FineWeb | +0.000144 | [−0.000100, +0.000493] |
| Agreement | 64 | WikiText | −0.001551 | [−0.004161, +0.000287] |
| Agreement | 64 | TinyStories | +0.000350 | [−0.000299, +0.001328] |

The equal-panel paired difference is **+0.00150** for the bracket head
with interval **[+0.00052, +0.00270]**: on this declared mixture,
weight-only has lower KL as well as much better bracket-path fidelity.
For agreement, the difference is **−0.000077**, interval
**[−0.000532, +0.000298]**; the aggregate KL ranking is unresolved.
The two cells where corpus PCA has a lower sample mean both have
intervals crossing zero. TinyStories bracket differences are heavy-tailed:
corpus PCA beats weight on more than half of individual windows but has
substantially worse *mean* KL. This is why both paired context results and
tail behavior matter.

The result does **not** establish that data-aware reduction is inferior.
Corpus PCA minimizes unweighted head-state reconstruction on a small
FineWeb fit set, which is neither the downstream KL objective nor the
targeted finite intervention path. A larger or better specified
data-aware objective—such as output-weighted state error or directly
estimated downstream KL sensitivity—could improve the natural-text
frontier. The panel bootstrap treats windows as samples and does not
account for possible source-document overlap between the two length
panels. The rank-four interventions also change only one selected head
at the final position, so the KL values should not be read as a whole
model compression result.

Reproduce the model evaluation with
`flagship_later_head_rank4_kl_scale.py` and paired analysis with
`analyze_later_head_rank4_kl_scale.py`. Per-window KL and flips are in
`full_width_results/later_head_rank4_kl_scale.json`; all panel
differences and confidence intervals are in
`full_width_results/later_head_rank4_kl_scale_summary.json`.
