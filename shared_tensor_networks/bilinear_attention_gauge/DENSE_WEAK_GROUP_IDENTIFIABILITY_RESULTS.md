# Shared dictionary ambiguity: output sparsity versus reusable input rules

The [dense-coded attention experiment](DENSE_WEAK_SHARED_ATOM_RESULTS.md)
showed that an edit-sparse rotation of a faithful rank-four function recovers
the designed overlapping four-bit graph. We now ask whether a more general
shared dictionary with sparse group membership would pick that same graph.
The answer on this toy is **no without an intervention or input-structure
prior**. This is an exact same-function counterexample, not an optimization
failure.

## Two exact decompositions

For each trained model and reduction measure, write the rank-four
query/output-by-key matrix as `W = C B`. `B` has four unit-norm dictionary
atoms over the 16 key tokens; `C` has eight query/output rows. Nonorthogonal
atoms are allowed. The membership graph is the support of `C`, called active
when a row coefficient reaches 10% of that row's largest coefficient.

An **output-row dictionary** takes the four ordinary-query output rows of
`W` as atoms. It is constructed solely from the learned function, with no
task-bit code. A **raw-bit oracle dictionary** uses the four known token-bit
patterns; it is an evaluation/counterexample, not an available featurizer.
We project candidate atoms into the exact rank-four rowspace, normalize,
then solve the coefficient matrix by pseudoinverse. Both decompositions
reproduce *the same fixed W* to less than `3.2e-13` maximum absolute error
over all ten seeds and four reduction measures. Thus all task predictions,
logits, and function-level interventions are identical before any component
deletion. Neither representation is a more faithful reduction.

The output-row basis is not numerically pathological: its mean dictionary
condition number is 1.86, versus 1.00 for the raw-bit oracle. Its atoms
overlap (mean maximum pairwise cosine .445), so this is a genuine allowed
nonorthogonal shared dictionary.

| Same rank-four function; 40 cuts | Mean active row–atom links | Mean normalized row-coefficient L1 | Raw-bit edit purity | Intended graph-cell accuracy |
|---|---:|---:|---:|---:|
| Output-row dictionary | **8.00** | **8.08** | .898 | .688 |
| Raw-bit oracle | 18.00 | 10.42 | **1.000** | **1.000** |
| SVD gauge | 25.00 | 12.93 | .653 | .719 |
| Fitted all-pair edit L1 | 18.78 | 10.60 | .994 | .976 |
| Fitted same-raw-D-bit edit L1 | 19.03 | 10.62 | .994 | .968 |

Every output-row decomposition has eight active links, one per nonzero
query/output row. Eight is the lower bound for this 10%-thresholded link
count, so a prior that *only* minimizes that support would prefer the
output-specific basis over the intended reusable-bit graph. Row-normalized
coefficient L1 points in the same direction. The exact coefficient support
can include tiny cross-links due to imperfectly identical trained query
rows; the active-link statement uses the declared 10% threshold.

The raw-bit oracle has more row links because A, B, and C each reuse two
bits, while D reuses three. Its atoms are far more selective under single
raw-bit token edits. The fitted edit-sparse gauges get close to that
selectivity and recover almost all the intended graph without receiving the
raw-bit oracle. Edit purity uses known bit-flip families **after fitting**;
it is a validation measure, not an extra training signal for the all-pair
gauge. The same-raw-D-bit prior was fit without direct D-flip edges, but
still sees the weight-derived function and mixed D information in other
edits.

## Why this matters for canonicalization

An invertible four-by-four coordinate change transforms `C B` into another
exact four-atom factorization. Allowing nonorthogonal atoms enlarges that
freedom. A structural prior is therefore part of any *interpretive*
canonicalization, even when the external attention function is known
exactly. Sparse group membership answers “which output combinations can
be represented with few links?” It need not answer “which computations are
reused across interventions?” This toy makes those objectives disagree at
the same rank, parameter count, and function error.

This does not show that edit purity identifies the model's physical heads
or unique internal computation. The oracle bit edits are available in the
synthetic grammar; analogous intervention families in a language model
would need independent construction and held-out validation. Also, these
are static query/output membership groups, not a trained sample-dependent
gate architecture. A learned overlapping-group featurizer should be
tested next on varied task compositions, preferably with unknown group
count and entire edit families withheld. Its stopping criterion is whether
it improves held-out group/intervention transfer over the simpler exact
edit-sparse basis at matched function error. This counterexample rules out
plain link sparsity as the sole proposed prior.

Reproduce with `dense_weak_group_identifiability.py`. All 200 factorization
rows and summary statistics are saved in
`dense_weak_group_identifiability.json`. The script asserts exact
reconstruction for every factorization and uses the saved ten trained
checkpoints' fixed rank-four cuts.
