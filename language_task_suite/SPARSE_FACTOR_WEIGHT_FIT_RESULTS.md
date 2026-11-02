# Weight-only sparse factor fitting, then language distillation

This follow-up tests whether the sparse-factor student's earlier fidelity gap
was due to its factorization or the initial language-loss/magnitude-pruning
schedule. It uses the same frozen two-layer varied-language teacher and the
same 29,491 surviving attention-edge budget as
[the architecture pilot](SPARSE_DEPTH_DISTILLATION_PILOT.md). One seed only.

## Data-free matrix fit

[`sparse_factor_weight_fit.py`](sparse_factor_weight_fit.py) begins with an
exact balanced-SVD factorization of each of the teacher's 12 attention
projections. It prunes globally through 70%, 50%, 40%, and 30% of the factor
weights, then runs 250 masked Adam steps at each budget to minimize mean
*relative* Frobenius reconstruction error over the 12 projection matrices.
It balances each factor's intermediate-coordinate scales every 20 steps.
No generated language or activations enter this stage.

| Surviving factor edges | Mean relative matrix error | Teacher KL on 256 independent varied-language streams |
|---:|---:|---:|
| 68,813 | 0.0006 | <0.00001 |
| 49,152 | 0.0306 | 0.00013 |
| 39,322 | 0.0642 | 0.00086 |
| 29,491 | 0.1847 | 0.01043 |

The effective language function remains close at 40% physical factor density,
despite nonzero error in individual matrices. At 30%, a weight-only fit no
longer reaches the fidelity of the original flat student at the same budget.

## Fixed-mask language recovery

Starting from the **data-free 29,491-edge mask**, we trained all weights for
460 steps of teacher KL with no L1, enforcing that mask. This matches the
number of language-loss updates used by the earlier function-pruned student
(80 L1 + 80 no-L1 before its fixed-mask checkpoint, then 300 no-L1), although
the weight-only route also has extra matrix-fit computation. The two runs
therefore compare routes to a sparse factorization, not equal total FLOPs.

| Route, 29,491 attention edges | Teacher KL, ordinary / return-shift / surface-shift |
|---|---|
| Flat student, prior fixed-mask recovery | 0.00087 / 0.00095 / 0.00130 |
| Function-pruned factors, prior recovery | 0.00312 / 0.00355 / 0.00405 |
| **Weight-fit factors → language recovery** | **0.00142 / 0.00136 / 0.00177** |

The data-free matrix fit materially improves the factor route under the same
physical edge budget. Its checkpoint has median intermediate fan-in and
fan-out of 19 each; 12 of 768 intermediate coordinates have a zero read or
write side. The flat sparse projection rows have median fan-in 39 at this
budget. These are local graph statistics, not human interpretability scores.

On a separate 512-stream panel containing **7,426 valid causal edits**, the
effect-L1 errors below compare each model's change in next-token distribution
with the exact generator's change; smaller is better.

| Model | Lexical verb | Cue | Verb induction | Previous-place skip | Nuisance determiner |
|---|---:|---:|---:|---:|---:|
| Original dense teacher | 0.273 | 0.215 | 0.244 | 0.219 | 0.013 |
| Flat sparse, recovered | 0.286 | 0.224 | **0.251** | 0.233 | 0.015 |
| Function-pruned factor, recovered | 0.293 | 0.240 | 0.276 | 0.246 | 0.019 |
| **Weight-fit factor, recovered** | **0.274** | 0.231 | 0.255 | **0.232** | **0.014** |

The weight-fit factor is competitive with the flat sparse student on these
known rules, though slightly worse on cue and induction. The larger edit panel
supports near-parity of *behavioral response*, not recovery of a readable
circuit or transfer to an unseen rule family. The teacher itself is imperfect
relative to the exact generator.

## Are the intermediate coordinates interpretable?

[`audit_sparse_factor_units.py`](audit_sparse_factor_units.py) tests the eight
highest-norm first-layer coordinates in each of the six attention projections.
On 64 different streams (922 valid edits), it zeros each coordinate at every
position, recomputes the model's response to each edit, and records token
preferences, fan-in/out, and rule-specific causal impact. A similarly selected
set of flat projection output channels is a reference, though a flat output
channel is not mathematically the same object as a factor intermediate unit.

The largest factor coordinates are causally important but broad. For example,
ablating intermediate coordinate 0 in the first block's output projection
changes the model's response by mean L1 **1.436** for lexical edits, **0.855**
for cue edits, **0.607** for induction edits, and **0.603** for previous-place
edits. The top first-layer token responses in Q/K/V factors also mix people,
places, and structural tokens. We have not found a clean rule-specific unit
among these largest coordinates. The audit covers only top-norm first-layer
units and does not rule out more specific smaller units or multi-unit circuits.

Thus this route has achieved a useful **faithful sparse reparameterization**;
the evidence for a more interpretable computation is still missing. A next
test of the same method should use a teacher trained on both verb and object
rules, then ask whether factors remain faithful across both task classes and
whether held-out task-specific interventions identify reusable, localized
units. If not, stop treating physical factor sparsity as sufficient and add
an explicit causal-specificity or path-level objective.

The two-site follow-up is now reported in
[V3_SPARSE_FACTOR_TRANSFER_RESULTS.md](V3_SPARSE_FACTOR_TRANSFER_RESULTS.md):
the factor route remains behaviorally close on verb and object edits, while
task-selective coordinates also exist in the flat sparse control.

Raw outputs: [weight-only fit](sparse_factor_weight_fit_pilot/),
[language recovery and unit audit](sparse_factor_weight_then_distill_pilot/),
[flat and function-pruned controls](sparse_architecture_recovery_pilot/).
