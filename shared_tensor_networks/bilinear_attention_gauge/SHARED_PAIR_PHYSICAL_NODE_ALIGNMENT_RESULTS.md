# Readable product nodes do not match the trained head-pair computations

The [dense-token shared-product toy](DENSE_CODED_SHARED_PAIR_RESULTS.md)
has a successful **function-level** story. Ten independently trained
three-head models solve three count bits and two parity outputs,
including a rare whole-combination holdout. A weight-derived rank-three
ordinary-function cut retains 100% five-output accuracy in all ten
seeds. A joint-nullspace gauge recovers three readable product nodes
corresponding to the A–B, B–C, and A–C relations; B–C is reused in
two query/output cells.

This experiment asks a stricter question: does one of those recovered
nodes correspond to a **physical pair of attention heads** feeding the
model's learned quadratic readout?

## Exact comparison

Write the learned three-bit response of physical head $h$ as
$z_h(q,c)=H_h(q)c$, where $c$ is the vector of eight key-token counts
and $q$ is one of two query tokens. The parity readout uses a learned
symmetric matrix $Q_p$ for output $p$. Expanding
$(\sum_h z_h)^\top Q_p(\sum_h z_h)$ produces six unordered physical
head-pair quadratic forms: three self-pairs and three cross-pairs.
Their sum is the model's exact quadratic count function. The script
checks this identity against the trained weights in float64.

Separately, each fixed rank-three function cut has six monomials in
its rotated key dictionary. We take its three strongest nodes by the
earlier label-free coefficient-energy rule. Each node defines another
exact quadratic form indexed by parity output, query, and two token
counts. The script checks that all six semantic forms sum to the
rank-three function's quadratic form. It then compares the strongest
three with (i) the best *single* physical head-pair form and (ii) the
**entire six-form physical span**. This is generous to the physical
candidate: a semantic node may use an arbitrary linear combination
of all six head-pair terms, not merely one deletion.

The metric is uniform Euclidean norm over all coefficients of the
$[2\ \mathrm{parity},2\ \mathrm{query},8\ \mathrm{key},8\ \mathrm{key}]$
form. It is an exact weight-only comparison, not a sample of language
sentences. The earlier rare-combination task score checks usefulness
on the grammar; it does not enter this alignment score.

## Twenty trained models, two token codings

Means across the three strongest nodes in each of ten seeds per
coding, using the joint-nullspace gauge:

| Coding | Mean node best-single-pair cosine | Mean fraction of node norm **outside all six** physical pair forms | Mean largest query/output-cell energy share: semantic node / physical pair | Cancellation ratio of all six semantic outside-span parts |
|---|---:|---:|---:|---:|
| Paired tokens | .458 | **.749** | .887 / .555 | .018 |
| Dense overlapping tokens | .451 | **.744** | .900 / .592 | .008 |

Every one of the 60 strongest nullspace-gauge nodes has a physical-span
residual of at least **.564** of its coefficient norm. The best
single-pair cosine never exceeds .726 in the dense models or .668 in
the paired models. Thus no strong semantic node is equal to one
physical head-pair contribution, and even an arbitrary linear
combination of the six physical contributions cannot reproduce it as
a polynomial over all count vectors.

At the same time, the six semantic forms' **sum** is close to the sum
of the physical forms: mean full-versus-rank-three quadratic-form
relative error is .011 for dense tokens and .025 for paired tokens.
The outside-physical-span parts of the individual semantic forms
mostly cancel in that sum; the final table column is the norm of their
sum divided by the sum of their norms. This is a concrete example of
a legible external decomposition whose individual pieces do not
correspond to that physical node partition.

The prior changes the *legibility* of the same retained function, not
its physical alignment. On dense coding, the SVD gauge gives .717 mean
single-cell energy share, key-entry L1 gives .521, and the
joint-nullspace gauge gives **.900**. Their mean physical-span
residuals remain .736, .718, and .744, respectively. A much more
readable semantic product graph can therefore become no more
identifiable with physical head-pair operations.

## Implication

The weight-only reduction and sparse/nullspace rotation are useful:
they find a compact, transferable **function-level computation**.
Calling its product nodes the model's actual reusable physical
computations would be unsupported. Deleting a semantic monomial is a
counterfactual in the chosen explanation; deleting a physical head,
head-pair term, or QK branch is a different intervention. The
[branch-edit red team](SHARED_PAIR_BRANCH_CAUSAL_RESULTS.md) separately
shows that the ordinary rank-three space misses much of the physical
QK-branch response, even at a small edit dose.

This result is specific to the six head-pair terms of this trained
quadratic readout. It does not rule out other physical intermediate
nodes, other groupings, or equivalence on a restricted data
distribution. The next comparison should ask which semantic-node
counterfactuals are approximately realizable **on task data**, under
a distribution-weighted metric and a constrained set of actual
internal patches. A positive match there would be weaker than
polynomial identity but more relevant to the model's ordinary uses.

Reproduce with shared_pair_physical_node_alignment.py. Its JSON
contains every seed, coding, gauge, selected node, exact-sum gate,
best physical-pair cosine, physical-span residual, and cancellation
score. No new model training or GPU run is required.
