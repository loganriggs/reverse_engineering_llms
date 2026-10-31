# Intervention geometry across 36 heads: stable axes need edit coverage

The learned [toy shared-dictionary comparison](LEARNED_OVERLAPPING_GROUP_RESULTS.md)
found that an edit-response prior, rather than group-link sparsity alone,
selected reusable input rules. We tested a more principled version of that
prior on the 546M model. Two previously captured, disjoint 800-edit panels
contain ten input-substitution types and native final-position responses
from all nine heads in layers 9, 11, 13, and 15. We fitted a basis to
**all 36 heads**, without using answer labels or downstream gradients.
The input-edit type identities do define the intervention prior.

## Mathematical motivation and implementation

Suppose edit-response coordinates have an invertible shared mixing matrix
`A`, and
each intervention family `t` changes independent latent components with
covariance `D_t` diagonal. Then `C_t = A D_t Aᵀ`. If a pooled covariance is
positive definite, whitening converts these to symmetric matrices with a
common orthogonal eigenbasis. If the vectors of eigenvalues across families
are distinct for every pair of components, a generic linear combination
has simple spectrum and identifies the common axes up to sign and
permutation. This supplies a **conditional** reason to prefer an
intervention-defined gauge. The trained model need not obey this exact
independence assumption.

Our empirical implementation is narrower than that theorem. For each
head, we fix the top four training-response PCA directions, then rotate
**orthogonally within that fixed subspace** to minimize the sum of squared
off-diagonal entries in ten edit-type covariance matrices. One fit gives
equal trace to each type; another retains relative response energy.
Eight Givens-rotation starts are tried. The method passes a synthetic
common-eigenbasis recovery check, and all 72 real fits terminate
successfully. We independently fit the same type-defined objective on
the validation panel to measure axis stability. No reduced full-model
forward pass is asserted by this input-response test.

| Across 36 heads | Equal-type fit | Raw-energy fit |
|---|---:|---:|
| Median held-out off-diagonal energy / PCA | .463 | .292 |
| Heads improved over PCA | 36/36 | 34/36 |
| Median train/validation matched-axis cosine | .988 | .982 |
| Mean rank-four validation response-energy coverage | .380 | .380 |

The geometry objective generalizes to a second panel of the *same ten edit
types*. This is not a whole-task discovery result: token substitution
identities can repeat across panels, and the edit families are supplied.
More importantly, the average rank-four cut discards **62%** of native
*head-state* edit-response energy. Its median coverage is only .319; just
four of 36 heads exceed .75 on the validation panel. This is a warning,
not a downstream-fidelity verdict: the learned output projection may
suppress some discarded directions. The subsequent
[all-head rank frontier](ALL_HEAD_RANK_FIDELITY_FRONTIER_RESULTS.md) checks
that immediate output response and each edit family separately.

The four heads selected *only by training-panel rank-four coverage above
.75* are 11/5, 13/3, 13/5, and 13/8. Their independent validation
coverages are .792, .843, .785, and .936. The existing
[bracket](LATER_HEAD_UNLABELED_INTERVENTION_RESULTS.md),
[secondary-head](SECONDARY_HEAD_FINITE_RESULTS.md), and
[punctuation](PUNCTUATION_HEAD_FINITE_RESULTS.md) finite-path tests
already identify 13/8 as a strong but known bracket locus and 13/3 and
13/5 as punctuation-responsive heads with small downstream paths. Thus
this scan does not select a new high-coverage, strong-output candidate.
In the bracket head 13/8, the raw-energy joint fit even has **1.142×**
PCA's held-out off-diagonal cost; joint diagonalization is not universally
better on the heads that pass the fidelity gate.

## Whole-edit-type exclusion

For those four training-selected heads, we refitted each gauge ten times,
omitting one entire input-edit type each time. We evaluated how much of
that withheld type's validation covariance its **best axis in hindsight**
captured. This oracle-axis compactness does not choose a useful component
without seeing the withheld type; it is a representation audit.

Across 40 head/type holdouts, mean oracle-axis fraction is .466 for either
joint fit, compared with .443 for PCA. The overall leave-one-type-out
basis remains close to the full basis (mean matched-axis cosine .985 for
equal-type, .988 for raw-energy), but the specialized axis can depend
strongly on the omitted family. For head 13/8, the equal-type full fit
places **99.6%** of bracket-edit energy on one axis; with bracket edits
excluded, the best axis retains **65.8%**. For punctuation head 13/3,
the raw-energy full fit gives **54.5%** punctuation concentration versus
**41.1%** after excluding punctuation; ordinary PCA gives 54.1%.
High whole-basis alignment therefore does not certify transfer of an
unseen specialist.

## Decision

Intervention covariance is a reproducible coordinate prior when a type
has support in fitting data, consistent with the toy result. It does not
solve the two harder problems: choosing a cut that preserves the
**relevant immediate-output edit families**, and predicting a new
downstream behavior when its edit family was absent. More rotation tuning
on these already studied bracket and
punctuation heads would duplicate prior work. The next real-model test
should either improve stage-one coverage with a weight-only or mixed
function/usage reduction before gauge fitting, or freeze a novel output
prediction from a task-blind scan and test finite paths on new token
identities and documents. The toy-side next gate remains a trained
multi-composition model with sample-dependent groups and a whole-family
holdout.

Reproduce with `flagship_joint_edit_geometry.py` and
`flagship_joint_edit_type_holdout.py`. JSON metrics and NPZ rotations are
in `full_width_results/all_head_joint_edit_geometry*` and
`all_head_joint_edit_type_holdout.json`.
