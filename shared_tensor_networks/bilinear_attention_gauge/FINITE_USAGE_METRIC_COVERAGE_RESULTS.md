# Finite corpus size changes a tight cut; edit fidelity still chooses the needed rank

We tested how a finite unlabeled corpus changes the *measure* used to reduce
an already trained attention function. Ten independently trained three-head
bilinear-attention toys compute three count-sign bits and a quadratic parity
output. Each model is competent on a rare query/composition withheld from
its supervised training. We extracted its exact combined `6×8`
query/output-by-key-token function from the weights. For each model and
sample size `8, 32, 128, 512, 2,048, 8,192`, five independent unlabeled
training-distribution sequence samples supplied query/key occurrence weights.
We fitted rank-two and rank-three tables under empirical weights, a 5%
uniform floor, and a 50% uniform mixture. A uniform weight-only SVD is the
fixed baseline. The same independent 30,000-sequence test per model scores
behavior only **after** fitting; no answer labels enter any cut.

The samples determine only the **weights of the fitting objective**. Every
fit evaluates the model's complete weight-derived token function, and the
weighted solver starts from its full-table SVD. This is how a dataset-aware
*reduction of known weights* works, not a data-only attempt to learn the
function from eight examples. At `N=8`, 11/50 panels had no rare-query
sequence and the mean number of covered rare-query key cells was 4.44/8;
at `N=32` it was 7.52/8, and all eight cells appeared in every `N=128`
panel. Missing cells still had a weight-derived initialization.

| Rank-two cut | Rare held-out all-four accuracy, N=8 | Rare held-out accuracy, N=8,192 | Broad relative table MSE, N=8,192 |
|---|---:|---:|---:|
| Uniform weight-only | 10.0% | 10.0% | 0.3154 |
| Empirical usage | 20.3% | **53.5%** | 0.3234 |
| 5% uniform floor | **26.6%** | 51.8% | 0.3230 |
| 50% uniform mixture | 15.9% | 18.8% | 0.3173 |

The moderate mixture improves broad table fidelity but substantially lowers
rare-task accuracy at this insufficient rank. At `N=8,192`, its rare accuracy
minus pure empirical is `-34.7` percentage points, with a model-seed
clustered bootstrap 95% interval of `[-59.5, -12.0]` points. At `N=8`,
the same difference is only `-4.4` points, interval `[-13.4, +4.2]`; a
small uniform floor helps some severely undercovered panels. The empirical
rank-two rare accuracy rises from 20.3% at `N=8` to about 52–54% at
`N=512–8,192`, but more samples do not remove its rank bottleneck. These
intervals reflect ten trained model seeds with five data replicates each;
they do not establish a universal ordering of priors.

At rank three, all three dataset-weighted measures retain at least
**99.64%** rare-held-out accuracy in every one of their 900 fitted panels,
and uniform weight-only achieves 100% in all ten models. To choose rank
without seeing answers, we compared each cut's complete token-edit
response with the weight-derived original on all 28 key-token swaps,
separately for ordinary and rare query. The maximum per-query relative
edit error is at least `0.276` for every rank-two cut, but at most
`0.0052` for every rank-three cut. Predeclared error thresholds of
`0.01` or `0.05` therefore select **rank three in all 1,800
model/sample/measure/threshold decisions**. The selection panel is
answer-free but includes both query forms and all tokens; it is not a
whole-rule discovery holdout.

This separates three issues. More usage data reduces measure noise.
A uniform floor protects broad function fidelity but can trade away
rare-task accuracy when rank is too tight. An answer-free edit-fidelity
gate detects the missing direction and chooses sufficient rank here.
The clean success at rank three is partly due to full weight-table access
and a strong SVD start; it does not show that eight text examples reveal
the rare rule. The next informative toy must make a competent rare rule
weak in weight/edit energy or include a nuisance that competes for rank,
then withhold that rule's edit family from *selection* and test whether
weight-only, data-aware, and mixed cuts still retain it.

Run `three_count_finite_metric_coverage.py` for the 1,820 cut records
(ten uniform baselines per rank plus 1,800 weighted fits), then
`analyze_three_count_finite_metric_coverage.py` for model-clustered
intervals and answer-free rank decisions. Full results are in the
corresponding JSON files.
