# A task-faithful canonical gauge exposes reusable product structure

**Physical-edit follow-up:** [separate QK-branch tests](SHARED_PAIR_BRANCH_CAUSAL_RESULTS.md)
show that the task-faithful rank-three reduction does not preserve the
branch counterfactuals; this is independent of the exact gauge.

**Dense-token red team:** [overlapping A/B/C token attributes](DENSE_CODED_SHARED_PAIR_RESULTS.md)
show that key-entry sparsity is coding-dependent. A condition-wise
product prior and a constructive joint-nullspace gauge recover the
reusable function-level graph in both codings.

The earlier [one-node pruning study](THREE_COUNT_PRUNING_BUDGET_RESULTS.md)
could not separate graph simplification from information lost in a rank-two
cut. This experiment trains a new three-head bilinear-attention model from
random initialization on **five outputs**: the three A/B/C count-sign
bits, plus `A XOR B` and `B XOR C` of the model's ordered bit outputs.
The two parity outputs share a variable. The rare query rotates the bit
order from A/B/C to B/C/A, so the full function needs three conditional
product edges: A-B, B-C, and A-C. B-C is reused across query types.

Ten models train on 100,000 variable-count sentences each. The rare-query
`A=0, B=1, C=0` combination is excluded from train and validation; each
model is evaluated on 30,000 independent sentences. All ten native models
reach **100% five-output accuracy** on the full test and rare holdout.
The exact first-layer `6 × 8` query/output-by-key table plus two learned
quadratic readouts reproduces direct model logits with maximum absolute
error `6.9e-13`.

## A reduction that preserves the task

| Shared-key cut | Full-test five-output accuracy | Rare-holdout accuracy | Uniform table MSE |
|---|---:|---:|---:|
| Native | 1.000 | 1.000 | 0 |
| Uniform rank 2 | .570 | .000 | .311 |
| Frequency rank 2 | .507 | .229 | .316 |
| Uniform rank 3 | **1.000** | **1.000** | .0004 |
| Frequency rank 3 | **1.000** | **1.000** | .0006 |

Uniform rank three is a weight-only SVD of the complete token table.
Frequency rank three uses 20,000 unlabeled training-law sentences. Both
retain the five-output task in all ten models. The gauge tests below
rotate *within* each fixed rank-three subspace and therefore leave its
unpruned function exactly unchanged.

## Exact-gauge legibility

For a three-coordinate key basis `D`, the induced quadratic readout has
six possible monomials: three squares and three pairwise products. The
key-L1 gauge minimizes the absolute entries of `D` without task labels.
The graph variants additionally reward concentrating the learned
quadratic coefficient energy in two or three monomial nodes; they also
see no task labels. Known A/B/C token contrasts are used **only after
gauge fitting** to score axis alignment and intended edges.

| Uniform rank-three gauge | Matched variable cosine ↑ | Energy in top three monomials ↑ | Energy on expected query-specific product edges ↑ |
|---|---:|---:|---:|
| SVD | .918 | .790 | .626 |
| Random rotation mean | .837 | .707 | — |
| Key L1 | **.9997** | .983 | .978 |
| Key L1 + three-node graph cost (`λ=3`) | .9996 | **.987** | **.982** |

The frequency rank-three cut gives almost identical results: key L1
alignment .9997 and top-three energy .982; the three-node graph variant
has alignment .9995 and top-three energy .987. In all 20 model-by-cut
cases, the three strongest nodes after key L1 or the three-node graph
fit are precisely the three cross-product coordinates. A fixed
**two-node** graph prior is misspecified for this query-conditional task;
its top-two energy is only about .81 even in the semantic gauge. This
shows why graph size should be selected or conditioned on query, not
assumed from the common-query examples alone.

Keeping only those three weight-selected product nodes, while retaining
the full linear and constant readout terms, gives:

| Uniform rank-three gauge | Mean full-test five-output accuracy | Mean rare-holdout accuracy | Mean parity-output relative MSE |
|---|---:|---:|---:|
| SVD | .9263 | .8875 | .174 |
| Key L1 | **1.0000** | **1.0000** | .013 |
| Three-node graph (`λ=3`) | **1.0000** | **1.0000** | **.010** |

Under the frequency cut, key L1 gives .99999 full and .99947 rare
five-output accuracy after pruning; the three-node graph gives 1.00000
on both. These are genuine task checks on independent sentences, not
only coefficient sparsity. The learned readouts, however, were trained
with task labels; only the subsequent reduction and gauge fitting are
label-free.

Finally, delete each matched product node separately and measure its
effect on the two parity outputs under each query type. Under the
uniform cut, the key-L1 gauge places **.994** of A-B node response
energy in the expected common-query first parity, **.998** of B-C in
the two expected cells (common-query second and rare-query first), and
**.986** of A-C in rare-query second. The corresponding SVD shares are
.784, .889, and .635. These fractions are computed with equal weight
per query/output cell, so rare-query effects are visible despite their
lower natural frequency. Frequency-cut numbers are nearly the same.

This is strong evidence for a compact, reusable **function-level**
description of this trained toy. It does **not** prove the three physical
attention heads implement those nodes independently, or that a
function-preserving gauge can identify the actual internal DAG. The
grammar and token pairs make key sparsity a well-matched prior. A next
test should change the token coding or task family, then assess whether
the same no-label gauge and node-selection rule still find the rule.
The 546M model requires full-context interventions because isolated
pair labels previously reversed downstream.

Reproduce with `train_shared_pair_count.py` for seeds 0–9,
`analyze_shared_pair_count.py`, `shared_pair_gauge.py`,
`shared_pair_graph_prune.py`, and `shared_pair_node_interventions.py`.
Their JSON files and all ten checkpoints are saved here.
