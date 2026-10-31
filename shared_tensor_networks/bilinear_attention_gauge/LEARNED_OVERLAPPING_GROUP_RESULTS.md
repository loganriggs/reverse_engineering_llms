# Learned overlapping groups: the edit prior works, but adds little over rotation

The [exact dictionary counterexample](DENSE_WEAK_GROUP_IDENTIFIABILITY_RESULTS.md)
showed that a sparse output-to-atom graph can prefer output-specific atoms
over reusable input rules. We now **learn** a four-atom, nonorthogonal shared
dictionary for each fixed rank-four `8 query/output × 16 key` attention
function. Its eight rows are static groups that may select overlapping
subsets of four atoms. This is a controlled static group factorization,
not a sample-dependent block-sparse autoencoder or a computational DAG.

## Fit and safeguards

Write the fixed function as `W = C B`, where `B` contains four unit-norm
token-response atoms and `C` contains eight group coefficient rows. An
invertible four-by-four coordinate transform parameterizes both, so every
candidate represents *exactly the same W*. We optimize the sum of
row-normalized coefficient L1 and `ratio ×` atom edit-response L1 across
all 120 token pairs. A barrier penalizes dictionary condition numbers above
four. The 10%-of-row-maximum support of `C` defines active group links.
The exact reconstruction assertion passes for every saved fit with maximum
absolute error below `5.7e-14`. No task labels or known bit identities
enter fitting; the bit graph is used only afterward for evaluation.

The first two model seeds were a **development panel**. They suggested an
edit-weight ratio of 80. We then evaluated that fixed ratio on the other
eight seeds under two reduction measures, without selecting by their task
labels. The full ratio sweep is exploratory. Five-start results revealed
local minima, so we additionally reran the eight held-out seeds at ratio
80 with 32 starts, matching the restart count of the earlier orthogonal
rotation. Finally, we used the earlier **label-free** edit-sparse rotation
as a warm start for the group model, and compared that pipeline separately.

| Held-out seeds 2–9; fixed ratio 80 | Active links | Graph-cell accuracy | 64-label one-atom new-task accuracy | 64-label one-pair XOR accuracy |
|---|---:|---:|---:|---:|
| Weight-only, group-link L1 alone | 8.0 | .688 | .939 | .886 |
| Weight-only, edit + group L1, 5 starts | 16.25 | .945 | .984 | .969 |
| Weight-only, edit + group L1, 32 starts | 18.25 | .992 | .992 | .985 |
| Empirical usage, group-link L1 alone | 8.0 | .688 | .939 | .886 |
| Empirical usage, edit + group L1, 5 starts | 15.25 | .914 | .957 | .914 |
| Empirical usage, edit + group L1, 32 starts | 18.0 | 1.000 | 1.000 | 1.000 |
| Either measure, edit-gauge warm start + group L1 | 18.0 | 1.000 | 1.000 | 1.000 |

The warm-start result covers all ten seeds as well as the eight held-out
seeds; the table reports the held-out panel. The eight-link solution is the
theoretical minimum active-link count for eight nonzero rows, but its atoms
track mixed output features. As the edit prior gains weight, the learned
dictionary accepts more overlapping memberships and approaches the four
reusable raw-bit atoms. The ratio-80 warm-start fits have mean dictionary
condition number near one and do not exploit nearly dependent atoms.

The 32-start weight-only run retains one local solution on seed 3:
its objective is `97.71`, while the warm-start fit reaches about `96.00`
with exact graph recovery. Thus the poorer result is an optimizer failure,
not evidence that the objective prefers the wrong graph on that seed.
The empirical 32-start run finds the intended graph in all eight held-out
seeds. All 100 full-sweep fits and all 16 held-out 32-start fits report
optimizer success; local minima remain possible despite successful
termination. The edit-gauge warm start supplies a reliable practical
pipeline on this toy, but it inherits the earlier rotation's input prior.

## Does the learned group model add interpretive value?

It verifies that a **nonorthogonal shared dictionary with overlapping
groups** can recover the designed reusable structure when edit-response
geometry is explicitly favored. Pure group-link sparsity cannot. However,
the simpler exact orthogonal edit-sparse gauge already reaches about
99.99% one-atom and one-pair transfer on the *same* held-out task panel;
the warm-start group's perfect result is too small an absolute gain to
justify more tuning on this toy. The group model is static: there are no
per-example activation gates, no hierarchy, and no recovered physical
head-level computational DAG. We stop further scalar penalty sweeps here.

The next high-information test is either (1) a trained multi-composition
toy with sample-dependent group activation and a whole composition family
withheld from fitting and selection, or (2) the 546M head with a fixed
edit-defined group prior evaluated on disjoint documents and downstream
interventions. The go/no-go criterion is a material gain in held-out
causal or new-task transfer over the exact rotation at matched response
fidelity, not simply a lower sparsity penalty.

Reproduce with `fit_dense_overlapping_groups.py`. Saved artifacts include
the two-seed pilot, ten-seed ratio sweep, eight-held-out-seed 32-start
check, and ten-seed edit-warm run. Each JSON records objective values,
conditioning, exact reconstruction, graph scores, and few-shot transfer;
companion NPZ files contain the fitted `B` and `C` factors.
Running the script with `--check` verifies its autograd objective against
finite differences, including an ill-conditioned coordinate transform,
and checks exact factor reconstruction.
