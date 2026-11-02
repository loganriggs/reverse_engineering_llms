# A functional quotient compresses two final readout paths

The [prior bond experiment](V3_RULE_HOLDOUT_BOND_RESULTS.md) found that object
induction reaches the last-token readout through both the incoming residual
state and the final attention output. This test asks whether a compact basis
should represent those paths separately or represent **their combined effect
on the readout**. It uses the same frozen, two-layer bilinear transformer
family and exact-oracle, grammar-valid verb/object edits.

Let `h` be the residual entering the final attention block, `z` its
pre-output attention activation, and `W` its output matrix. With no MLP in
this block, the next-token distribution is

`p = softmax(LM_head(final_norm(h + W z)))`.

For an original/edited pair, the readout only receives
`Δr = Δh + W Δz`. The linear map `(Δh, Δz) ↦ Δr` has a kernel of at least
64 dimensions in this 64-wide model: many path-level changes are invisible
once combined. This is a **local functional quotient**. It is exact for this
final readout, including its nonlinear normalization and softmax because
both act *after* the sum. It is not a canonical form for the whole attention
network; upstream computations and the decomposition into `h` and `z` remain
distinct. An orthonormal basis within any selected subspace also retains a
rotation gauge.

## Controlled featurizer comparison

Two independently trained teachers each supply 120 fit edits per one of
eight families and 60 per family on development streams. **Verb and object
induction edits are entirely absent from fit and development.** The test
panels are new streams with 160 examples per family, including both withheld
induction families. Identical edit triples are removed across splits; the
paired final test panels have zero removals. Both teachers did learn all
rules in their original language training, so this is a *featurizer-label*
holdout, not a whole-task model-training holdout or new-task discovery.

At the same total coordinate budget, compare:

- **Combined:** rank `k` projector on `Δr` in the 64-dimensional final
  residual space.
- **Separated:** rank `k/2` projector on `Δh` plus rank `k/2` on `Δz`, then
  combine their projected contributions through `W`.

For each representation, fit orthonormal bases by equal-family mean squared
error on either the teacher's probability *effect* or the generator's exact
oracle effect. A separate development panel selects the optimization step.
Baselines include edit-contrast PCA, context PCA, 4,096 randomly sampled
ordinary-stream residual states without edit labels, and a weight-only LM-head
SVD. The task-blind ordinary streams can contain the held-out rule in
unlabeled form; they are a different information condition from the labeled
fit arms. Every projected intervention starts from the original readout and
adds a projected edit delta. Original, edited, and full-width recombination
replay the teacher within 1.3e-6 maximum probability error.

The table reports mean L1 error between the compact intervention's next-token
probability **change** and the teacher's full edit effect. Lower is better.
The rank-16 combined and separated learned bases each have 1,024 unconstrained
matrix entries before orthogonality; rank-32 arms each have 2,048. Rank-16 and
rank-32 arms were evaluated on the **same fresh test panel for each teacher**.

| Total coordinates / method | Teacher 0 verb induction | Teacher 0 object induction | Teacher 1 verb induction | Teacher 1 object induction |
|---|---:|---:|---:|---:|
| 16, combined teacher-effect fit | .074 | **.064** | .058 | **.063** |
| 16, separated teacher-effect fit, 8+8 | .172 | .153 | .203 | .173 |
| 16, combined contrast PCA | .076 | .221 | .082 | .255 |
| 16, task-blind stream PCA | .146 | .265 | .127 | .243 |
| 16, combined oracle-effect fit | .106 | .103 | .114 | .086 |
| 32, combined teacher-effect fit | **.003** | **.004** | **.006** | **.005** |
| 32, separated teacher-effect fit, 16+16 | .077 | .057 | .077 | .067 |
| 32, combined contrast PCA | .011 | .039 | .021 | .054 |
| 32, task-blind stream PCA | .019 | .090 | .028 | .162 |

The teacher-effect-trained combined projector is the strongest compact
method on the held-out object rule at both budgets. Rank 32 captures nearly
all native effects for both rules and both teachers, whereas path-separated
16+16 projectors still leave .057–.077 L1 error. The combined rank-16
projector also outperforms a path-separated 8+8 projector at a matched total
coordinate and trainable-matrix budget. This is evidence that eliminating
readout-invisible path differences is more efficient *at this site* than
retaining path identity. It does not prove that path identity is unimportant
for explaining how the computation is built upstream.

The oracle-effect objective is a useful negative result. It uses exact
generator labels on the eight **seen** edit families, yet transfers less
well than teacher-effect fitting to both withheld induction families.
On the paired rank-16 panels, teacher-effect fitting has object-induction
oracle-effect L1 .386/.406; oracle-effect fitting has .408/.412. The native
teachers themselves have .365/.380. Supervision from known rules does not
automatically correct an unseen rule, and a projector can preserve a teacher
effect while remaining less correct than the teacher against the oracle.

For a local necessity/sufficiency check, removing the rank-32 combined
projected delta from the full donor delta leaves only .003/.004 effect L1
for teacher 0 verb/object induction and .006/.005 for teacher 1. Applying
only the projected delta gives similarly small errors from the native effect.
This is strong **last-readout** mediation; it is not evidence that individual
basis axes are semantic units or that intervening in the actual running model
at earlier layers has the same causal effect. A whole-model compression is
not yet built: unedited predictions are unchanged by construction, and the
projection is applied to *edit deltas* using original/donor activation pairs.
Ordinary model KL and task-prediction accuracy therefore remain separate
future tests, not metrics established by this intervention experiment.

Next, rotate or sparsify the saved combined projector without changing its
subspace, then test whether individual axes or small groups correspond to
specific rules under sufficiency, necessity, and spillover checks. Compare
that with a path-aware representation that explicitly pays for preserving
`h` and `z` separately; better behavior fidelity alone cannot decide which
description helps humans. Use a genuinely withheld rule from featurizer
training and an independent language-model training holdout before claiming
task discovery. Broader surface forms and FineWeb-distribution checks are
still required for natural-language relevance.

Reproduce with [`eval_v3_two_path_holdout.py`](eval_v3_two_path_holdout.py).
Paired rank-16 and rank-32 JSON records and reusable basis tensors are in
[`results/`](results/) with names
`v3_two_path_holdout_rank{16,32}_modelseed{0,1}_paired330{01,11}.{json,pt}`.
