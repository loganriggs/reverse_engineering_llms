# Sparse factors on a two-site verb-and-object language

The [weight-first sparse-factor route](SPARSE_FACTOR_WEIGHT_FIT_RESULTS.md)
nearly preserved a two-layer teacher on the earlier varied-verb language. We
then applied the same method to the independently trained
[`surface_mixture_v3_object8`](surface_mixture_v3_object8/) teacher, whose
visible-prefix generator mixes four competing mechanisms at **verb** slots
and four at **object** slots. This broadens the tested tasks and lets us ask
whether the new intermediate coordinates distinguish the two rule families.
It is still controlled synthetic language, not a claim about FineWeb realism
or newly discovered tasks.

The v3 teacher has two bilinear attention layers, width 64, four heads, and
a 313-token vocabulary. Its held-out exact-oracle KL is about **0.064** over
all positions, **0.096** at verb positions, and **0.101** at object positions.
It is competent but imperfect. The current eight-object training lexicon is
deliberately small; when the generator expands to sixteen objects, the
teacher's object-slot oracle KL rises above **6**. That shift tests whether a
student preserves the teacher's behavior, **not** whether either model
generalizes the correct object rule to new objects.

## Matched sparse students

The factor route first fits each teacher projection with two sparse matrices
using only weights: 250 masked optimization steps after each global prune to
68,813, 49,152, 39,322, and **29,491** attention edges. At 29,491 edges, its
mean relative matrix error is **0.189** and its initial teacher KL is **0.0167**.
It then receives 460 no-L1 language-distillation steps at the fixed mask. A
flat two-layer control is pruned directly to the same edge budget and receives
80 L1, 80 no-L1, then 300 more no-L1 steps: 460 language-loss updates in all.
Both retain dense embeddings and LM heads; only attention weights are sparse.
The weight-only route has extra matrix-fit compute, so this is an edge-matched
comparison, not an equal-FLOP optimizer comparison. The balanced-SVD start and
masked fitting select one sparse factor basis; this is not a unique canonical
form of the teacher's function.

| 29,491 attention edges | Teacher KL, ordinary v3 panel | Teacher KL, expanded-object panel | Exact-oracle KL at verb / object slots |
|---|---:|---:|---:|
| Flat sparse | **0.00123** | **0.00402** | **0.102 / 0.107** |
| Weight-fit factor, then distill | 0.00140 | 0.00501 | 0.104 / **0.105** |

These 256-stream panels are disjoint from training. Factorization remains
close to the flat control on both known rule sites at the same physical edge
budget, with a small aggregate teacher-KL gap. A second distillation data seed
with the **same factor mask** gives ordinary-panel teacher KL **0.00155**; it
does not test stability across independently fitted factor bases. Its base
rule-edit responses on the unit-validation panel are likewise close to the
first recovery seed.

On a separate 512-stream panel with **12,251 valid edits**, lower effect-L1
error means a closer match to the exact generator's change in next-token
probabilities. Representative results:

| Causal edit | Teacher | Flat sparse | Sparse factor |
|---|---:|---:|---:|
| Lexical verb | 0.277 | **0.289** | 0.299 |
| Verb induction | 0.312 | **0.316** | 0.318 |
| Previous-place verb | 0.255 | **0.262** | 0.266 |
| Object adjective | 0.238 | **0.247** | 0.250 |
| Object induction | 0.369 | 0.372 | **0.371** |
| Previous-event object | 0.215 | **0.218** | 0.223 |
| Nuisance object place | 0.046 | **0.047** | 0.055 |

The factor model is close on both task classes, with a clearer gap on the
nuisance-object-place edit. This is **behavioral and causal-effect fidelity**,
not proof that its physical intermediate units are readable or reusable.

## Intermediate-unit audit and a flat-basis control

[`audit_sparse_factor_units.py`](audit_sparse_factor_units.py) ablates one
first-layer projection coordinate at all positions and measures its effect on
valid verb and object edits. From a 64-stream discovery panel we selected a
factor coordinate in block 0, K2, index 1. On an independent **12,100-edit**
panel, its normalized mean effect impact is **1.157** for verb edits and
**0.279** for object edits, a **4.14×** verb/object ratio. Ablation increases
mean exact-oracle effect error by **0.174** over verb rules versus **0.010**
over object rules. A second distillation data seed, starting from the same
weight-fit mask, gives a **4.18×** ratio on the same held-out panel. This is
real task-family selectivity conditional on that factor basis and teacher.

The largest output-factor coordinate is broad instead: normalized impact is
**1.94** on verb rules and **2.50** on object rules. The selective K2 coordinate
also has **36 read connections and 23 write connections**, and its highest
token responses mix structural words, places, and people. It is a causal
subtask handle, not yet a simple human-readable concept.

Crucially, a scan of the flat sparse model's projection output channels also
found verb-selective channels. Three candidates chosen on a separate 24-stream
discovery panel retained **5.2–5.4×** normalized verb/object ratios on the
same 12,100-edit validation panel. A factor intermediate coordinate and a
flat projection output channel are different mathematical objects, and the
factor K2 unit has greater absolute effect impact. Still, **task-family
selectivity is not unique to the sparse factorization**. We should not call
this an interpretability improvement without a matched circuit-level test.

The next test should optimize an explicit path or causal-specificity objective
while preserving KL, then evaluate whether a small set of factor units and
edges is sufficient and necessary for held-out verb and object effects. It
should include whole-rule-family training holdouts if we want to claim new-task
discovery. This two-site replication meets the stopping criterion for the
single-verb sparsity detour: physical sparsity alone yielded fidelity and a
selective handle, but not a uniquely legible or recovered circuit.

Scripts and raw outputs: [matrix fit](sparse_factor_weight_fit.py),
[distillation](distill_sparse_depth_vs_factor.py),
[causal edits](eval_sparse_architecture_causal_edits.py),
[unit audit](audit_sparse_factor_units.py),
[v3 weight fit](v3_object8_sparse_factor_weight_fit/),
[v3 students and edits](v3_object8_sparse_recovery/), and
[second factor recovery seed](v3_object8_sparse_recovery_seed1/).

The main runs can be reproduced from the repository root with the local
PyTorch environment:

```bash
PY=/home/loganriggs/Coding/neuron_circuits/.venv/bin/python
$PY language_task_suite/sparse_factor_weight_fit.py \
  --teacher-dir language_task_suite/surface_mixture_v3_object8 \
  --output-dir language_task_suite/v3_object8_sparse_factor_weight_fit \
  --edge-budgets 68813,49152,39322,29491 --steps-per-budget 250 \
  --eval-streams 256 --length 72
$PY language_task_suite/distill_sparse_depth_vs_factor.py \
  --teacher-dir language_task_suite/surface_mixture_v3_object8 \
  --output-dir language_task_suite/v3_object8_sparse_recovery \
  --arms factor \
  --recover-from-dir language_task_suite/v3_object8_sparse_factor_weight_fit \
  --recover-budget 29491 --recover-steps 460 \
  --batch 16 --length 72 --eval-streams 256
```
