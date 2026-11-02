# Short-to-long section lookup in the joint language

The [v8 section-scoped task](SURFACE_MIXTURE_V7_SCOPE_RESULTS.md) has a
verified prefix oracle, grammar-valid section-key and color edits, and a
clear failure: both a three-layer bilinear model and a same-shell softmax
control trained directly on 256-token mixed streams learned a color shortcut
while barely responding to the queried section number. This experiment
tests a **same-vocabulary, same-architecture curriculum** before claiming
that the task exceeds these models' capacity.

## Short capability stage

The [short generator](pretrain_v8_scope_lookup.py) samples one supported
starter clause from v8, then forces two numbered sections with distinct badge
colors. Both section queries from the same document are supervised. A real
sample from the independent test stream is:

> Afterward, Henry checks the valuable badge near the library before the delivery, while Aaron watches.
>
> Section 3: home. Detail: the badge was red.
>
> Section 4: garden. Detail: the badge was green.
>
> What color was the badge in section 3? Answer: **red**

The alternate query asks for section 4 and must answer **green**. Section IDs,
topics, colors and the starter clause vary. The [independent audit](results/v8_scope_short_pretrain_audit.json)
checked 1,000 worlds, all 2,000 answer prefixes and every sampled-token
transition for exact v8 support. It independently reconstructed the two
visible section facts, verified distinct answers and changed answer under
the key edit. Prefixes span 49–69 tokens. This short distribution is a
*curriculum*, not a natural web-text sample or a substitute for the full
v8 mixed-rule process.

The pretrainer uses the **same 368-token vocabulary and three-layer,
width-64, four-head RMSNorm bilinear architecture** as the scratch v8 model.
It learns only answer-token CE, batch 32 worlds (64 queries), for 1,500
updates. Development CE selects the checkpoint. At 750 steps it is still
key-blind (50.3% accuracy, key-edit TV .0085); at 1,000 it reaches 97.8%
accuracy and TV .939. The selected 1,500-step checkpoint reaches **100%**
accuracy, answer CE **.00037** and key-edit TV **.99968** on 512 independent
test worlds. The query-blind baseline is 50% and TV 0. This verifies the
architecture can learn keyed section lookup in the short format. It does
not show zero-shot transfer: without joint fine-tuning the same checkpoint
gets only **37%** accuracy on 200 long v8 questions, below their 46.5%
query-blind reference.

## Joint mixed-language transfer

We initialized the existing 256-token v8 trainer from the selected short
checkpoint and ran **1,200 joint updates**. It uses the same generator
parameters, three-layer architecture, batch 32, learning rate .003, ordinary
next-token CE, scope-answer weight 2, counterfactual weight 1, seed and
joint-training stream as the existing 1,200-step scratch checkpoint. The
only intentional joint-stage change is initialization; the curriculum arm
also consumed 1,500 short answer-only updates before that stage. Thus this
is a training-schedule comparison, **not** matched token exposure or compute.
There is one training seed and two fresh evaluation-stream seeds.

On two independent 200-question panels, each model is scored on the **same**
prefixes and supported edits. The key-blind Bayes accuracies are 45.4% and
43.7%; the exact oracle changes answer with TV 1 for a section-key edit and
TV 0 for an irrelevant-color edit.

| Model | Accuracy, panels A/B | Answer CE, A/B | Section-key edit TV, A/B | Irrelevant-color TV, A/B |
|---|---:|---:|---:|---:|
| Scratch, 1,200 joint updates | .455 / .400 | .945 / .997 | .0012 / .0012 | .542 / .528 |
| Short curriculum + 1,200 joint | **.775 / .750** | **.477 / .574** | **.627 / .593** | **.294 / .298** |

The curriculum model has learned substantial section-key use, unlike
scratch. It is not exact: its section-key effect L1 error from the oracle is
**1.073/1.128**, and changing an irrelevant section's color still moves its
answer distribution by about .30 TV. Relevant-color edit TV is .745/.716
(oracle 1). The answer rate cannot by itself certify scoped retrieval; the
key and irrelevant edits are the decisive diagnostics.

The short stage trained on exactly two sections. A separate fresh panel
breaks the long-stream result down by the number of completed sections:

| Completed sections | Curriculum accuracy, panels A/B | Key-blind accuracy | Curriculum key-edit TV, A/B |
|---|---:|---:|---:|
| 2 (`n=200/221`) | .835 / .846 | .500 | .606 / .670 |
| 3 (`n=86/67`) | .605 / .672 | .333 | .466 / .506 |
| 4 (`n=14/12`) | .214 / .500 | .250 | .382 / .444 |

This is some transfer to three-section cases, with much weaker and noisy
evidence for four. The random section IDs and colors do not create a fixed
ID→color lookup, but the short stage did **not** withhold section–color
combinations or entire surface forms. That stronger generalization remains
untested.

The joint-task tradeoff is large. On two **identical fresh 256-stream v8
panels**, lower exact-oracle KL is better:

| Rule site | Scratch KL, panels A/B | Curriculum KL, panels A/B |
|---|---:|---:|
| Section answer | .912 / .899 | **.506 / .504** |
| Verb | **.582 / .581** | .911 / .921 |
| Object | **1.124 / 1.127** | 1.761 / 1.777 |
| Object-place answer | **4.348 / 4.206** | 5.140 / 4.839 |
| Source answer | **3.045 / 3.247** | 4.458 / 5.003 |

The curriculum solves part of a previously failed binding while worsening
the other joint rules. This **does not** provide a competent all-rule teacher
for featurizer discovery. It does show that the earlier v8 scratch and
softmax failures were not evidence that the three-layer bilinear model
cannot express section-key lookup. The task-rich v8 surface is still rigid
and much more frequent than similar dependencies in short web windows; the
[v8 realism audit](SURFACE_MIXTURE_V7_SCOPE_RESULTS.md) found essentially no
scope-answer sites at its low-rate setting. These experiments establish
controlled learning and interference, not FineWeb realism.

## Decision gate

Do not decompose this checkpoint into a purported section circuit yet. A
strong joint teacher should retain key-edit sensitivity while driving
irrelevant-color TV much closer to zero and restoring verb/object/source
competence on a balanced panel. The next training control should either
continue the curriculum at a lower learning rate with explicit task-balanced
supervision or train scratch for a comparable *joint* update budget; both
must use fresh causal panels and report token exposure. A second training
seed is needed before treating the curriculum gain as reliable. If the
joint model remains selective only for section questions, use the competent
short model as an isolated capability control and return featurizer work to
already competent multi-rule checkpoints. The featurizer deliverable would
be a whole-rule-family holdout comparison of weight-only, data-aware and
causal sparse bases, scored for reconstruction, teacher KL, exact-oracle
prediction, key edits and irrelevant edits separately.

Reproduce with [`pretrain_v8_scope_lookup.py`](pretrain_v8_scope_lookup.py)
and [`audit_v8_scope_short_pretrain.py`](audit_v8_scope_short_pretrain.py),
then `train_surface_mixture.py --generator v8 --init-checkpoint` with the
saved joint configuration. The [pretraining artifacts](results/v8_scope_short_pretrain_seed1/)
and [joint artifacts](results/v8_scope_curriculum_seed1/) include checkpoints
and development records. Fresh evaluations use
[`eval_surface_mixture_v7_scope.py`](eval_surface_mixture_v7_scope.py),
[`eval_v8_scope_cardinality.py`](eval_v8_scope_cardinality.py), and
[`eval_v8_joint_rule_kl.py`](eval_v8_joint_rule_kl.py).
