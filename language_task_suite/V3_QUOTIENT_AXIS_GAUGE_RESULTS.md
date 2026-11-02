# Sparse axes inside a faithful readout subspace

The [two-path quotient](V3_TWO_PATH_FUNCTIONAL_QUOTIENT_RESULTS.md) provides
a rank-32 subspace of the final residual whose projected edit deltas preserve
the native teacher's verb- and object-induction effects to L1 .003–.006.
The subspace is useful, but its individual axes are arbitrary: replacing its
orthonormal basis `U` by `U R` for any orthogonal `R` leaves `U Uᵀ` and the
full 32-dimensional intervention unchanged. This experiment tests which
**gauge prior** makes fewer axes useful at a time.

Both independently trained two-layer bilinear teachers use the v3 verb/object
toy language. The rotation-fit panel has 120 valid edits per each of eight
known rule/control families, with a separate 60-per-family development panel.
Verb induction and object induction are absent from rotation fitting and
development. Two fresh test panels contain 160 examples per family. Exact
duplicate edit triples across splits were removed; none occurred in these
panels. The underlying rank-32 subspace was already fitted to the same eight
known families, so this is **rotation transfer**, not discovery of an
untrained task.

We compare the saved orientation, a random orthogonal rotation, PCA of the
edit coordinates, a rotation minimizing normalized L1 of **edit activation
coordinates**, and a rotation minimizing normalized L1 of the rows of the
LM-head weight matrix restricted to the subspace. The activation objective
uses edit-selected inputs but no rule labels in its loss; the balanced panel
construction is task informed. The output-weight objective uses no examples
at all after the subspace is fixed. All five rotations preserve the rank-32
projector to maximum elementwise difference below 3e-7 and reproduce the
original/edited teacher readout within 1.3e-6 probability error.

For an edit, keep its four largest-magnitude coordinates in a chosen gauge,
apply only those directions at the final readout, and compare the resulting
next-token probability *change* to the native teacher's. This top-four
selection is **per example** and uses the donor edit activation. Lower L1 is
better; full 32-coordinate fidelity is .003–.006.

| Gauge, top four axes per example | Teacher 0 verb induction | Teacher 0 object induction | Teacher 1 verb induction | Teacher 1 object induction |
|---|---:|---:|---:|---:|
| Saved orientation | .194 | .249 | .192 | .217 |
| Random rotation | .220 | .233 | .241 | .229 |
| Coordinate PCA | .197 | .251 | .187 | .216 |
| **Edit-activation sparse** | **.124** | **.153** | **.128** | **.124** |
| Output-weight sparse | .197 | .234 | .184 | .222 |

The activation-sparse rotation reduces the fit coordinate L1/L2 score from
3.99/4.05 in the saved orientations to 3.09/3.10. On held-out induction
edits its four largest axes carry about 77–80% of squared coordinate energy,
versus 56–59% for the saved orientation. This translates to lower
top-four causal-effect error on both teachers. The output-weight rotation
successfully lowers its own objective—output-row L1/L2 falls from about
4.5 to 2.18/2.22—but gives little or no gain in top-four edit fidelity.
Weight sparsity and intervention sparsity select different gauges here.

Per-example sparsity is not yet a reusable circuit. To test a fixed group,
we used **32 labeled calibration edits per withheld family** to choose four
axes by mean coordinate energy, repeated the choice on an independent
32-edit calibration panel, then evaluated the first group on new streams.
The table gives fixed-group effect error from the native teacher, followed
by the overlap of the two calibration selections (Jaccard). The same-gauge
per-example top-four errors are much lower. The fixed-group split removed
zero and four duplicate triples, respectively, for the two teachers.

| Gauge / family | Teacher 0 fixed-group L1 | Teacher 0 overlap | Teacher 1 fixed-group L1 | Teacher 1 overlap |
|---|---:|---:|---:|---:|
| Saved, verb induction | .271 | .60 | .322 | .33 |
| Sparse, verb induction | .250 | .60 | .304 | .33 |
| Saved, object induction | .322 | .14 | .244 | .60 |
| Sparse, object induction | **.225** | .33 | **.205** | .60 |

The sparse gauge improves this fixed-group object test, but even its best
four-axis groups remain far from full-subspace fidelity. The selected groups
are moderately unstable across two small calibration panels, and the
calibration uses held-out family labels, so this cannot count as unsupervised
task identification. It is consistent with **instance-dependent sparse
coordinates** rather than one four-axis module per rule. A different group
selection objective or larger groups could perform better; this test only
rejects the easy claim that a sparse rotation alone exposes a tiny fixed
induction circuit.

This is also a local counterfactual analysis, not a running sparse language
model. The top-k intervention requires the original and donor activations.
It does not establish ordinary next-token KL, full-model accuracy, or a
unique semantic decomposition. Orthogonal rotation preserves Euclidean
distances and any rotation-invariant clustering result exactly; it changes
axis-aligned sparsity and legibility, which is the hypothesis tested here.

The next featurizer experiment should allow **context-dependent gates over a
shared direction dictionary**, then ask whether small groups are sufficient
and necessary for held-out effects with low spillover. Keep the whole-rule
rotation/featurizer fit holdout and avoid using the labeled calibration
panels as evidence of discovery. In parallel, broaden task surfaces and
evaluate their dependency distribution against natural text before treating
this synthetic-language result as representative of LLMs.

Scripts: [`eval_v3_quotient_axis_gauges.py`](eval_v3_quotient_axis_gauges.py)
and [`eval_v3_fixed_axis_groups.py`](eval_v3_fixed_axis_groups.py). Raw JSON
and saved rotations are in [`results/`](results/) as
`v3_quotient_axis_gauges_modelseed{0,1}_test34{001,011}.{json,pt}`;
fixed-group records are `v3_fixed_axis_groups_modelseed{0,1}_test36{001,011}.json`.
