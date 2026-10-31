# Direct function-error budgets for simplifying a canonical attention circuit

The [independent-seed test](THREE_COUNT_GRAPH_PRIOR_FRESH_RESULTS.md) showed
that a saturated coefficient-count penalty is unreliable. Here the gauge
choice has an operational meaning: **among exact rotations of a fixed
rank-two attention function, choose the sparsest key dictionary for which
deleting one quadratic product node changes the function by at most a
declared relative MSE budget**. The deleted node is selected on the fit
measure without task labels. The first-layer rank reduction is held fixed;
gauge rotation alone has zero function error before node deletion.

We compare two fit measures on ten independent scratch-trained models
(seeds 10–19):

* **Complete uniform token domain:** enumerate all `8³` ordered key
  triples for each of two query types. No sentences are sampled. This is
  weight-only once the uniform domain is declared, though it is still a
  distributional choice.
* **Unlabeled context measure:** 4096 variable-count contexts from the
  training grammar, with no task labels used in gauge selection.

Both measures minimize key-entry L1 among gauges whose best one-node
deletion has fit relative MSE at most `ε ∈ {.01,.03,.1}`. If no gauge is
feasible, the script records this and uses the minimum-error gauge as a
diagnostic fallback. Evaluation uses exhaustive `8⁴` ordered key
quadruples, independent natural contexts, higher-count contexts, and
30,000 task sentences including a whole-combination holdout.

At `ε=.03`, the choice of fit measure changes what transfers. Entries
below are ten-seed means; the original key-L1 basis is a baseline with
its lowest-energy node selected under the named measure. The `.03` row is
illustrative from a three-budget sweep, not a preregistered estimate of
performance on new model families.

| Rank-two cut, fit measure | Designed-variable cosine ↑ | Uniform quadruple deletion MSE ↓ | Natural test deletion MSE ↓ | Higher-count deletion MSE ↓ |
|---|---:|---:|---:|---:|
| Uniform cut, uniform key L1 | **.947** | .092 | .068 | .069 |
| Uniform cut, uniform budget | .928 | **.026** | **.015** | **.015** |
| Uniform cut, natural budget | .935 | .067 | .023 | .024 |
| Frequency cut, uniform key L1 | **.980** | .041 | .066 | .066 |
| Frequency cut, uniform budget | .948 | **.021** | .033 | .034 |
| Frequency cut, natural budget | .939 | .030 | **.018** | **.019** |

Under the frequency cut, the uniform fit has better error on the
exhaustive unseen length-four domain, while the natural fit has better
error on independently sampled natural and higher-count contexts. This
is a real metric choice at the **legibility/simplification stage**, even
after the rank cut has already been chosen. At `ε=.03`, all ten uniform
fits are feasible for both cuts; natural fitting is feasible in all ten
uniform-cut models and nine of ten frequency-cut models. The infeasible
case uses the diagnostic minimum-error fallback, so it is not evidence
that the constraint was satisfied.

A stricter `.01` budget further lowers pruning error but tends to mix
the known variables: under the uniform cut, uniform-fit alignment drops
to .914, versus .947 for key L1. The original information-theoretic
ambiguity remains: the same exact reduced function can have a sparse
variable basis or a sparse post-pruning circuit, and neither uniquely
identifies the physical implementation.

Task accuracy is a separate, harsher check. At `ε=.03`, mean full-test
parity accuracy after pruning is .651 for the uniform-domain fit and .629
for the natural fit under the uniform rank-two cut; the unpruned cut is
.685. Under the frequency cut, both post-pruning fits are .597, versus
.604 unpruned. Crucially, the rare held-out four-output accuracy under
the **uniform** rank-two cut is zero for every gauge tested: its rank
reduction already loses ordinary bits that gauge selection cannot
restore. The frequency cut retains only about .27 rare four-output
accuracy after pruning. Lower quadratic MSE must not be read as task
recovery.

For a fixed gauge, the squared error from deleting node `j` is its
coefficient-weighted feature variance:

`Var_μ(c_j(q) φ_j(z)) / Var_μ(Σ_k c_k(q) φ_k(z))`, where
`φ = (z₀², √2 z₀z₁, z₁²)` and `μ` includes query and key context. Thus
the fit measure changes the moments of the monomial features, not merely
a cosmetic weighting of coefficients. Uniform enumeration computes these
moments exactly for its finite token domain; the context version estimates
them from unlabeled sentences. The observed transfer difference is
consistent with this moment dependence.

The next experiment needs a different compositional task with multiple
reused products, plus a higher-rank cut that preserves the task well
enough to isolate graph simplification from information lost at rank two.
Only then is transfer to the first-layer 546M model informative.

Reproduce with `three_count_pruning_budget.py` and
`three_count_pruning_budget_task.py`. Full per-seed results are in their
corresponding JSON files.
