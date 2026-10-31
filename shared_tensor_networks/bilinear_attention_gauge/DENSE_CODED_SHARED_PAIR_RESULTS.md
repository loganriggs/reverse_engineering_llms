# Dense token coding separates useful computation priors from token sparsity

**Physical-edit follow-up:** [QK-branch interventions](SHARED_PAIR_BRANCH_CAUSAL_RESULTS.md)
show that the ordinary rank-three semantic function space is not a
faithful description of internal branch counterfactuals, regardless of
which exact gauge is chosen within it.

The [paired-token shared-product toy](SHARED_PAIR_CANONICAL_RESULTS.md)
recovered three variables with key-entry L1, but its token code made each
variable a sparse pair contrast. We kept the same three-head bilinear
attention, two quadratic readouts, five outputs, training size, and rare
whole-combination holdout, while changing the eight key tokens to the
eight combinations of three binary attributes. Each token now contributes
to **all three** A/B/C count contrasts. The attributes are orthogonal in
the uniform eight-token metric but dense and overlapping in token space.
This is a new grammar, not merely a coordinate change applied to a fixed
trained checkpoint.

Ten models were trained from random initialization on 100,000 sentences
each. All achieve **100% five-output accuracy** on 30,000 independent
test sentences and on the rare held-out combination. Uniform and
unlabeled-frequency rank-three shared-key cuts also retain 100% of both
accuracies in every seed, so the gauge comparison is task-faithful.
Rank-two cuts fail substantially. The exact token-table plus learned
quadratic-readout reconstruction is checked against direct model outputs.

## The prior matters when token sparsity is false

Within each fixed rank-three cut, every orthogonal gauge computes the
same unpruned function. The **key L1** objective favors few nonzero token
entries. The **global three-node** objective additionally favors three
large monomials across all output/query conditions. A different,
**condition-wise one-node** objective favors one dominant quadratic
monomial *within each parity-output/query cell*, using only model weights
and the known output/query indexing. It does not see A/B/C labels or
sampled sentences. A constructive **joint-nullspace** gauge uses the
smallest-absolute-eigenvalue direction of each cell's induced quadratic
form; among four cells, it chooses three near-orthogonal null directions
with maximum volume and orthogonalizes them.

| Uniform rank-three gauge | Post-fit variable cosine ↑ | Mean one-node-per-cell energy ↑ | Five-output accuracy after keeping three nodes, full / rare ↑ | Correct-cell node-effect share ↑ |
|---|---:|---:|---:|---:|
| SVD | .907 | .647 | .828 / .763 | .735 |
| Key L1 | .700 | .393 | .688 / .632 | .224 |
| Key L1 + global three-node cost (`λ=3`) | .863 | .664 | .961 / .555 | .590 |
| Condition-wise one-node prior | **.999** | **.995** | **1.000 / 1.000** | **.998** |
| Joint nullspaces | **.9996** | .993 | **1.000 / 1.000** | **.998** |

The unlabeled-frequency rank-three cut gives the same qualitative
ordering: key L1 cosine .700 and rare accuracy .653 after pruning;
the global three-node cost gives .863 and .563; condition-wise and
nullspace gauges retain 1.000 rare accuracy and approximately .999 and
.9996 variable cosine. A condition-wise objective with a key-L1 term
also reaches 1.000 full and rare accuracy, so the result does not
require discarding key sparsity entirely.

For the condition-wise and nullspace gauges, the three strongest global
nodes are exactly the three cross products in all ten seeds under both
rank cuts. Deleting the matched A-B, B-C, or A-C node has its effect
almost entirely in the expected output/query cells. The B-C product
serves two cells, giving an explicit function-level example of reuse.
By contrast, the global coefficient-energy objective can look compact
while losing rare behavior: its mean rare five-output accuracy is only
.555 after the three-node simplification under the uniform cut.

The [paired-token control](SHARED_PAIR_CANONICAL_RESULTS.md) also remains
solved by these new priors. Its condition-wise gauge has mean variable
cosine .998, and its nullspace gauge .9984; three-node pruning retains
1.000 rare accuracy in all ten seeds under both rank cuts. Token-entry
sparsity happens to work there, but fails sharply when concepts overlap
in vocabulary space.

## Why nullspaces can identify this function-level basis

In the ideal three-coordinate basis, write the three distinct pair
forms as `H_AB = a(e_A e_Bᵀ + e_B e_Aᵀ)`,
`H_BC = b(e_B e_Cᵀ + e_C e_Bᵀ)`, and
`H_AC = c(e_A e_Cᵀ + e_C e_Aᵀ)`, with nonzero `a,b,c`.
Then `ker(H_AB)=span(e_C)`, `ker(H_BC)=span(e_A)`, and
`ker(H_AC)=span(e_B)`. Thus their *unordered* one-dimensional kernels
recover the three axes up to sign and permutation. A common orthogonal
gauge rotates these kernel lines together. The rare query repeats one
pair form, so selecting the three null vectors with maximum determinant
discards the duplicate without using task labels. This is an exact
identifiability statement for the ideal forms **within a fixed
Euclidean/orthogonal metric**, not for arbitrary GL gauges or distinct
physical arithmetic circuits.

The learned forms are approximate. Their selected near-null vectors
have minimum three-vector volume .996 in the dense-coded models, and
the constructed gauge achieves mean .9996 post-fit cosine, 1.000 rare
five-output accuracy after three-node pruning, and .998 correct-cell
deletion selectivity. Perturbation stability depends on the nonzero
eigenvalue gaps of the forms; near-degenerate forms could break this
construction. The held-out task checks here are still within one
synthetic task family, and physical QK-branch/head interventions have
not yet shown that the model implements the same internal graph.

This result changes the next real-model test. Token sparsity should be
treated as one prior, while *shared conditional computation* can be
tested by the family of downstream quadratic forms even when its
token-level directions are dense. For the 546M model we first need to
identify analogous condition-indexed bilinear forms and score their
full-context causal effects; pair-table labels alone previously failed
to transfer downstream.

Reproduce with `train_shared_pair_dense.py` for seeds 0–9, then
`analyze_shared_pair_dense.py`. That script runs the same rank,
exact-gauge, three-node pruning, and node-deletion checks used for the
paired-token control. Both grammars' JSON reports and checkpoints are
saved alongside these scripts. `check_shared_pair_nullspace.py` verifies
the ideal exact-form construction under 100 random orthogonal gauges.
