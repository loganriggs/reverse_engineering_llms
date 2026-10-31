# An unseen query exposes a limit of data-weighted reduction and sparse gauges

We reused ten trained two-head bilinear-attention models that answer two
independent A/B bit queries. The models were trained on both query forms;
only the **post-training reduction fit** was restricted here. Their exact
combined token functions have four query/output rows by six key tokens.
The ordinary query returns `(A,B)` and the swapped query returns `(B,A)`.
The dataset-only condition assigns zero fitting weight to all twelve
swapped-query cells. No answer labels enter any reduction or gauge fit.

We first construct two rank-two reductions that agree *exactly* on every
observed ordinary-query cell, so both have zero fitting error there:

- **Zero completion** sets the unseen swapped-query rows to zero. This is one
  possible data-only completion, not a claim that optimization must choose it.
- **Weight-informed completion** projects those unseen rows of the original
  model's exact weight-derived function into the same two-dimensional key
  subspace. This uses information unavailable from the ordinary-query data
  alone. It is a broad weight prior at the same rank and observed fit.

| Rank-two reduction, ten seeds | Ordinary-query accuracy | Swapped-query seen accuracy | Swapped-query held-out combination accuracy | Mean unseen-row table MSE |
|---|---:|---:|---:|---:|
| Zero completion | 100% | 30% | 10% | 35.88 |
| Weight-informed completion | 100% | 100% | 100% | 0.540 |
| Expected-frequency fit, swapped-query mass `0.0001` | 100% | 100% | 100% | 0.540 |
| Expected-frequency fit, swapped-query mass `0.05` | 100% | 100% | 100% | 0.512 |
| Uniform-query fit, mass `0.5` | 100% | 100% | 100% | 0.236 |

The positive-mass fits use the **exact expected pair weighting and the full
weight-derived token table**; they are not finite-sample estimates. Even a
tiny positive mass therefore supplies information about every swapped-query
cell, and this experiment does not establish that a finite corpus with that
frequency would suffice. At zero mass, the two completions are
observationally indistinguishable to the data objective. The weight-derived
table can still break the tie, but that is a prior supplied by the weights,
not evidence extracted from the restricted dataset.

Within either fixed rank-two function, we selected an exact orthogonal
gauge by key-weight L1 or all-15-pair edit-response L1. The latter's mean
alignment to the known A/B key variables is `0.9788` for zero completion
and `0.9783` for weight-informed completion. Key L1 is similarly close
(`0.9780` and `0.9777`). Thus a legible dictionary can recover the *key
variables* even when the reduction has no useful swapped-query mapping.
Rotating to sparse axes cannot restore a query/output row that the reduction
has discarded. This is a direct separation between legibility of a chosen
function and fidelity of the function itself.

The toy is intentionally small and its ordinary query already exposes both
A and B variables. It tests omission of a **query rule**, not discovery of
an entirely latent key feature. The next gate should fit from finite
unlabeled sequences with nuisance token changes, report coverage per
query/key cell, and withhold an output rule or composition from featurizer
selection. Compare data-only, weight-only, and mixed support priors before
testing sparsity, intervention signatures, and finite causal effects.

Reproduce with `dual_unseen_query_coverage.py`; all ten per-seed rows,
complete fit/holdout metrics, and exact-gauge scores are in
`dual_unseen_query_coverage.json`.

## Scoring audit

During this experiment, we found that `analyze_dual_output_language.score`
formed its `rare_seen` mask from an integer `swap` field. NumPy interpreted
the resulting integer array as row indices rather than a Boolean mask.
We corrected it and added `test_dual_score_masks.py`, which checks the
36 swapped-query seen cases and 12 held-out rare cases. This affected
**only** `rare_seen` fields from that scorer; full-task, common,
rare-held-out, table-error, variable-alignment, and component-removal
metrics use independent calculations. All affected saved dual-output
analyses were regenerated or rescored: joint metric/function/gauge runs,
the ten head-analysis, sparsity, and group runs, the alternate group
allowance, and eleven saved rank-one solutions. A recursive comparison
against the previous committed JSON confirms that every changed value
in those artifacts lies under a `rare_seen` key. The training histories
use a separate Torch Boolean mask and were not changed. The saved
rank-one solutions can be checked without retraining using
`repair_dual_rank1_rare_seen.py`.
