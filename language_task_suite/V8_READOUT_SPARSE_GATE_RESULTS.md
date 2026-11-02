# Are the section-response directions sparse per activation?

The [rank-16 key-edit basis](V8_SCOPE_COMPUTE_CONTROL_AND_READOUT.md) preserves
held-out section color effects well at the final combined residual. This
experiment asks whether a much smaller subset of its directions is active
for each example. It is a **fixed-dictionary gate baseline**, not yet a
trained overlapping-group or DAG featurizer.

We freeze the same v8 teacher and the rank-16 orthonormal dictionary fitted
to section-key edits. On two new exact-support test panels (seeds
48001/48002), each has 120 key edits and 49 relevant-color edits, plus
40/39 irrelevant-color edits. The fit panel (seed 46001) supplies only a
center for per-state coordinates and a ranking for fixed axes; no test edit
selects either. Original and edited output distributions replay from the
captured final residual to maximum error <1e-6.

At each budget `k`, we compare:

* **Fixed axes:** retain the `k` dictionary coordinates with largest mean
  squared key-edit delta on the fit panel.
* **State TopK:** encode original and edited residuals independently using
  their `k` largest centered dictionary coordinates, then subtract the
  sparse encodings. This is a per-activation gate available without observing
  the intervention difference.
* **Edit-delta TopK:** choose the `k` largest coordinates of the *difference*
  between edited and original residuals. This observes both worlds and is
  an intervention-aware ceiling, **not** a deployable activation featurizer.

Lower L1 error to the frozen teacher's native output effect is better. The
exact local 16-coordinate dictionary has error .037/.033 on relevant-color
edits; the full 64-dimensional residual would have zero error.

| Active coordinates | Fixed axes A/B | State TopK A/B | Edit-delta TopK A/B |
|---:|---:|---:|---:|
| 2 | .798 / .666 | .432 / .476 | **.243 / .282** |
| 4 | .347 / .386 | .214 / .150 | **.086 / .088** |
| 8 | .058 / .051 | .070 / .054 | .047 / .047 |
| 16 | .037 / .033 | .037 / .033 | .037 / .033 |

At four coordinates, adapting the gate to the activation improves over a
fixed global subset, but an edit-aware selector remains substantially
better. For section-key edits the corresponding four-coordinate errors are
.308/.216 fixed, .157/.090 state TopK, and .061/.065 edit TopK. For
irrelevant-color edits they are .160/.175, .155/.092 and .113/.067. The
eight-coordinate methods nearly recover the 16-dimensional dictionary's
teacher effect on both panels. State TopK gate Jaccard before/after a
relevant-color edit is .531 on panel A at `k=4`: the selected coordinates
often change, rather than forming an invariant four-axis module.

This supports a **context-dependent selector over a shared dictionary** as
the next featurizer experiment. It does not establish that the selected axes
are human-legible or causal units. In particular, the frozen teacher's own
effect L1 error from the exact oracle is .657/.495 for relevant-color and
.370/.400 for irrelevant-color edits. A perfectly faithful decomposition of
this teacher would reproduce those errors. A learned gate should therefore
be scored separately for teacher fidelity, exact-oracle fidelity, necessity
and irrelevant-edit spillover, with section–color combination holdouts.

Reproduce with [the evaluator](eval_v8_readout_sparse_gates.py). The
[first](results/v8_scope_sparse_gates_fresh48001.json) and
[second](results/v8_scope_sparse_gates_fresh48002.json) full score records
include all edit families, necessity/remaining effects, residual error and
gate overlap.

## First learned-gate pilot

We also tried a small [context gate](train_v8_readout_gate.py): a 16→32→16
network adjusts per-state coordinate scores, then a hard TopK mask selects
four directions. It starts exactly at state TopK. A straight-through
surrogate supplies gradients during training, but evaluation always uses the
hard four-coordinate mask. Fit and development use disjoint section-key
edits only; the same two fresh color-edit panels remain untouched.

The smoke test verified four active mask entries and a nonzero initial
gradient. On the full run (120 fit and 120 development key edits, 300
updates), the initial gradient norm was 1.84e-5. Development squared-effect
error was **.02193 at step 0** and worsened thereafter; development selection
therefore retained the initial gate. Relevant-color teacher-effect L1 on
the two test panels stayed **.214/.150**, identical to state TopK. This
pilot **fails the predeclared gate-improvement criterion**. It does not test
all learned selectors: the hard-mask surrogate, fit labels, and local
readout objective may all limit optimization. Do not tune this variant on
the held-out color results. The [run record](results/v8_scope_learned_gate_k4_seed0.json)
and selected [gate weights](results/v8_scope_learned_gate_k4_seed0.pt) are
saved for audit.
