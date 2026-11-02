# Three-task functional score modes in the joint language model

The [joint generator and capability study](V3_NESTED_JOINT_RESULTS.md)
established a three-block bilinear model that predicts contested verb and
object slots and nested-bracket closers. Its object-induction causal effect
remains weak, so this experiment asks a narrower question: does the
data-aware rank-8 first-layer token-score basis expose distinct *functional*
directions for the three task families that the model does implement?

The basis is fitted from 2,048 independent unlabeled joint streams (262,144
tokens). For each of four heads, one query-token and one key-token basis is
shared across all relative lags of the exact first-layer QK1×QK2 score
table. The two trained layers downstream of that first-layer table are
otherwise unchanged. Reconstructing the rank-8 projected table as a sum of
its modes has maximum absolute error 1.2e-6.

The [mode scan](surface_mixture_v3_nested_varlen_depth3/basis_mode_ablations_seed20261130.json)
checks all 64 head/side/axis removals on development data. It selects the
largest outer-opener edit damage, verb oracle-KL damage, object oracle-KL
damage, and a low-impact control. Selection is frozen before a separate
128-stream holdout and its grammar-valid verb, object, and bracket edits are
generated. Holdout counts include 549 lexical-verb, 283 verb-induction,
803 object-adjective, 302 object-induction, 124 outer-opener, and 66
inner-opener edits, plus nuisance controls.

| Removed rank-8 mode | Verb KL increase | Object KL increase | Outer-opener effect-L1 increase | Main held-out causal damage |
|---|---:|---:|---:|---|
| Head 3, query 0 (selected for outer close) | 0.001 | 0.010 | **0.654** | Outer closer edit L1 0.000 → 0.655; inner edit almost unchanged |
| Head 0, key 0 (selected for verbs) | **2.718** | 0.007 | ~0 | Lexical verb edit L1 0.285 → 1.728 |
| Head 2, query 0 (selected for objects) | 0.029 | **0.303** | 0.001 | Object adjective edit L1 0.263 → 0.563 |
| Head 2, key 3 (low-impact control) | ~0 | ~0 | ~0 | Representative edits remain near baseline |

The object mode also worsens object-induction edit L1 from 0.390 to 0.472,
but the native model already approximates that rule poorly. This should be
read as a useful object-family handle, not recovery of the intended
induction computation. The outer mode is more specific to *outer* closing
than to every bracket operation; inner-close edits barely change. The verb
mode is important to several verb rules, not a single semantic feature.

A [random-rotation red team](surface_mixture_v3_nested_varlen_depth3/mode_rotation_redteam_seed20261140.json)
holds the complete rank-8 score function fixed, randomly rotates the
selected eight-dimensional subspace, and compares the most damaging of all
eight axes in each rotation on a new 64-stream panel. Across 16 rotations,
the development-selected outer axis causes edit damage 0.499 versus at most
0.300 from any rotated axis; the verb axis causes KL damage 2.695 versus at
most 0.201. For objects, the selected axis causes KL damage 0.324, but the
largest rotated-axis damage reaches 1.009, and three of 16 rotations exceed
the selected axis. Thus this metric concentrates the outer and verb effects
well, while **object computation admits a more concentrated orientation**
within the same rank-8 subspace. Random axes were optimized only in the
limited sense of taking each rotation's maximum; the finite test is not a
canonical uniqueness theorem.

The next direct experiment is a task-specific axis search within these fixed
subspaces, with a penalty for off-task damage. Its fitting panel, axis
selection, and held-out validation must be disjoint. Success would mean
more causal concentration without changing the full rank-8 function; it
would establish a useful interpretability prior, not that the physical
network used that axis. Compare object, verb, and bracket effects, not only
the object result, and keep no-effect edits in the validation panel.

## A first search objective: prediction damage with off-task penalty

[`search_joint_task_axis.py`](search_joint_task_axis.py) tests 32 random
directions and 16 local refinements for each of the three previously
selected head/side subspaces. It maximizes target-task KL damage (or outer
opener-edit damage) minus positive damage to the other two tasks on a fresh
64-stream development panel, then fixes each direction before evaluating
128 new streams and all valid edit families. The intact rank-8 score table
is identical for every direction. The [search output](surface_mixture_v3_nested_varlen_depth3/task_axis_search_seed20261150.json)
shows holdout target/off-task damage:

| Task and head/side | Canonical target damage | Searched target damage | Largest off-task KL damage, searched |
|---|---:|---:|---:|
| Outer closer, head 3 query | 0.638 edit L1 | **0.961** edit L1 | 0.010 |
| Verb, head 0 key | 2.946 KL | **4.347** KL | 0.007 |
| Object, head 2 query | 0.322 KL | **0.883** KL | 0.008 |

The outer and verb searches also increase damage to the corresponding
held-out edit effects. For objects, however, more KL damage is **not** a
commensurate gain in causal legibility: object-adjective edit error rises
from 0.567 to 0.666 and object-induction edit error stays about 0.467,
despite object KL rising from 0.434 to 0.995. A direction can make object
predictions much worse without selectively disrupting the intended rule
effects. The objective therefore over-rewards distribution damage as a
proxy for computation. The next search must target multiple object-rule
edit effects directly, while penalizing nuisance and off-task effects, and
use another independent validation panel. Stop this axis search after that
comparison rather than tuning repeatedly on the same holdout.

## Causal-edit-targeted object axis: useful but still broad

[`search_joint_object_causal_axis.py`](search_joint_object_causal_axis.py)
keeps head 2's rank-8 query subspace fixed and compares the canonical axis,
the prior KL-selected axis, and a fresh 32-random-plus-16-local search. Its
development objective is the equal-weight mean increase in effect-L1 error
across object adjective, cue, induction, and previous-event edits, minus
positive nuisance object-place, verb-KL, and outer-bracket damage. The
search uses 32 development streams and then validates on 128 new streams.

On the [new holdout](surface_mixture_v3_nested_varlen_depth3/object_causal_axis_search_seed20261170.json),
the objective score is 0.089 for the canonical axis, 0.145 for the
KL-selected axis, and **0.178** for the causal-selected axis. The latter
raises mean damage to relevant object edits to 0.209, with off-task damage
0.010 and nuisance damage 0.021. However, its largest contribution is the
adjective edit: effect-L1 rises from 0.254 at the intact rank-8 baseline to
0.750 after ablation, versus 0.548 for the canonical axis. The induction
edit rises only from 0.385 to 0.472, versus 0.465 for the canonical axis.
Object oracle KL after ablation is **1.485**, versus 0.428 for the canonical
axis. This is a more object-selective destructive direction, but not a
clean four-rule module or evidence of reusable object computation.

Stop this local axis sweep. The comparison establishes that a data-aware
metric can expose task-selective axes and that additional direction search
can concentrate functional sensitivity, while neither KL damage nor mean
causal damage alone guarantees human-readable mechanisms. The next
featurizer test should use broader competent tasks and require simultaneous
prediction fidelity, each relevant causal effect, nuisance invariance,
held-out task-family transfer, and a compact sufficient/necessary set of
units or paths. Keep the joint model as a controlled benchmark, with its
object-induction weakness explicit; improve natural-language form separately.
This is one trained model seed on a deliberately artificial grammar; a
second independently trained joint model is needed before claiming that the
three-way mode split is a stable property of the architecture.
