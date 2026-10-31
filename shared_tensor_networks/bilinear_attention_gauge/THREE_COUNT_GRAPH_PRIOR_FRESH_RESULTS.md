# Independent-seed test weakens the quadratic graph-prior claim

The [first graph-prior sweep](THREE_COUNT_GRAPH_PRIOR_RESULTS.md) selected
`τ=.1, λ=.3` after inspecting ten scratch-trained models (seeds 0–9). We
froze those settings and trained ten **new** end-to-end attention plus
quadratic-readout models (seeds 10–19), using the same 100,000-sentence
training task and rare whole-combination holdout. We reproduced the
weight-only and unlabeled-frequency rank-two reductions, rotated within
each fixed subspace, chose the lowest-energy quadratic node on 4096
unlabeled fit contexts, and tested pruning on disjoint ordinary and
higher-key-count contexts. Task behavior uses 30,000 independent sentences
per model, including the held-out combination. All new native models have
at least .9982 rare-combination four-output accuracy.

| Reduction and gauge | Variable cosine ↑ | Held-out one-node deletion relative MSE ↓ | Shifted-count MSE ↓ | Full-test parity after pruning ↑ |
|---|---:|---:|---:|---:|
| Uniform, key L1 | **.947** | .064 | .065 | .630 |
| Uniform, fixed graph prior | .912 | **.029** | **.029** | **.653** |
| Frequency, key L1 | **.980** | .051 | .055 | .601 |
| Frequency, fixed graph prior | .979 | .049 | .053 | .602 |

This is a partial transfer. Under the uniform cut, the graph prior reduces
pruning MSE in seven of ten fresh seeds, but variable alignment drops by
.035 on average and by more than .1 in two seeds. Under the frequency cut,
the graph prior chooses *exactly the key-L1 gauge* in six of ten seeds;
the mean pruning improvement is only .002. The original frequency-cut
gain from .017 to .001 therefore does **not** replicate at the frozen
setting. The frequency-cut gauge still recovers the designed variables
well, but the graph penalty often adds nothing.

For context, the fresh rank-two reductions already damage the task:
unpruned mean full-test parity accuracy is .685 (uniform) or .604
(frequency), despite near-perfect native models. The graph-pruned
counterparts reach .653 and .602. Thus the representation-level pruning
improvement is not a solution to the more consequential reduction error.

The plausible failure mode is the saturated node-count surrogate: when
all three monomial strengths exceed `τ`, its graph cost is flat and cannot
select a different gauge; when a rotation does lower the cost, it can mix
the underlying key variables. We need a prior that accounts for the
**error of a simplified computation under a declared function measure**,
and tests whether that simplification preserves rare and intervention
behavior. This is more directly connected to the user's proposed
canonicalize-then-sparsify program than counting monomial coefficients.

Next, compare an explicit constrained objective such as “minimize key
sparsity subject to a held-out-calibrated error budget after pruning one
reused node,” evaluated on both uniform and corpus measures. Keep the
metric for the initial rank cut separate from this graph simplification.
Then test a *different* compositional task, since new seeds of the same
grammar do not establish mechanism transfer. No physical circuit recovery
is claimed.

Reproduce with `train_three_count_composed_scratch.py --seed 10` through
`--seed 19`, followed by `three_count_graph_prior_fresh.py`. Per-seed
measurements are in `three_count_graph_prior_fresh.json`; all ten
checkpoints and training histories are saved alongside it.
