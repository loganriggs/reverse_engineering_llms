# Edit-sparse gauge transfers to a second trained composition task

The [dense-code shared-product result](SHARED_PAIR_DENSE_EDIT_GAUGE_RESULTS.md)
showed that minimizing the L1 norm of all unlabeled token-swap responses
can recover three variables and a reusable function-level graph in ten
trained models. We tested the same prior on a related but distinct task:
ten **from-scratch** three-head attention models with three count-sign
outputs and one learned parity output. The rare query rotates bit order,
and an A=0/B=1/C=0 rare-query combination was withheld from training.
The models and their readouts were already trained before this gauge
experiment.

For each checkpoint, derive its exact `6×8` query/output-by-key token
table. Compare a rank-three **uniform weight-only SVD** with an
**unlabeled training-frequency weighted** rank-three fit. Both are
frozen before gauge fitting. Within each fixed function, compare the
SVD coordinates, minimum token-entry L1, and minimum L1 of differences
for all 28 unordered token pairs. The last objective sees neither
token attribute nor task labels. For each gauge, select the two
strongest quadratic monomials from the learned readout using global
coefficient energy, retain all linear and constant terms, and test
30,000 new sentences per model including the entire withheld
combination. The two-node budget is deliberately matched across
gauges; its size is not discovered.

| Rank-three reduction / gauge | Known-variable cosine | Energy in top two quadratic nodes | Four-output accuracy after two-node pruning: full / rare holdout |
|---|---:|---:|---:|
| Uniform / SVD | .862 | .610 | .878 / .798 |
| Uniform / token-entry L1 | **.99977** | **.978** | **1.000 / 1.000** |
| Uniform / all-pair edit L1 | **.99981** | .978 | **1.000 / 1.000** |
| Frequency / SVD | .877 | .613 | .889 / .807 |
| Frequency / token-entry L1 | **.99976** | **.978** | **1.000 / 1.000** |
| Frequency / all-pair edit L1 | **.99980** | .978 | **1.000 / 1.000** |

Numbers average ten independently trained models. All 20 unpruned
rank-three functions retain **100% full and rare four-output accuracy**;
every key-L1 and all-pair edit-L1 pruned model does too. The script
checks exact quadratic-monomial reconstruction and invariance of each
gauge's rank-three projector. The known bit contrasts enter only the
post-fit cosine score, never the gauge or node selection.

The edit prior therefore transfers across a change from two shared
parity outputs to one learned parity output, and across weight-only
versus frequency-weighted reductions. It is **not superior to key-entry
L1 here**: this task uses paired token coding, where individual token
entries already provide a good sparsity prior. Its distinct value is
that the same all-pair rule also worked on the earlier dense coding
where key-entry L1 failed. Both tasks still share the designed
three-bit count grammar, so this is not broad language-model
generalization or an identification of physical QK branches. The
all-pair vocabulary is tiny and enumerated exactly; scaling the
measure requires explicit coverage and context-noise controls.

Reproduce with `three_count_scratch_edit_gauge.py`; per-checkpoint
gauges and task scores are in `three_count_scratch_edit_gauge.json`.
