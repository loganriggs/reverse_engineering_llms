# V8 section curriculum: matched continuation and readout-basis controls

The [balanced curriculum result](V8_SCOPE_BALANCED_CONTINUATION.md) showed
section-key use alongside recovered joint-rule prediction. Two follow-ups
separate its learning result from a first local featurizer result. These are
controlled toy-language tests, not FineWeb prevalence estimates or a circuit
explanation of the larger model.

## Does the balancing stage itself teach section lookup?

We continued the original 1,200-step **scratch** v8 checkpoint for 1,200
updates using the *same* balancing objective, generator seed 101, batch,
learning rate and development-panel size as the curriculum-initialized
continuation. The curriculum arm alone had an additional 1,500 short,
two-section answer-only updates before its original 1,200 joint updates.
The two continuations therefore match their later 1,200-step stream and
supervision, but **not total training compute or data exposure**.

On identical fresh 256-stream panels (seeds 39101/39102), lower exact-oracle
KL is better:

| Rule site | Scratch + balance KL A/B | Curriculum + balance KL A/B |
|---|---:|---:|
| Verb | **.409 / .407** | .547 / .562 |
| Object | **.933 / .933** | 1.117 / 1.122 |
| Object-place answer | **3.900 / 3.902** | 3.947 / 4.033 |
| Source answer | **.878 / 1.109** | 2.104 / 2.288 |
| Section answer | .865 / .818 | **.268 / .246** |

On the same 200-question causal panels (seeds 39001/39002), scratch + balance
has section accuracy .515/.440, close to key-blind Bayes .456/.454, and
section-key edit TV **.00117/.00131** (oracle 1). Curriculum + balance has
accuracy **.885/.905** and key-edit TV **.810/.845**. The scratch continuation
still reacts to a relevant color edit (TV .511/.496), but almost equally to
an *irrelevant* section's color (TV .523/.517; oracle 0). The curriculum
continuation lowers irrelevant-color TV to .189/.181. Thus the balancing
stage repairs other rules from scratch but, at this compute point, does not
teach binding to the requested section. The curriculum benefit is specific
and the remaining cross-rule tradeoff is real. An independent curriculum
seed and a total-compute-matched scratch control are still needed.

## Do readout bases transfer across edits?

The [readout-basis runner](eval_v8_rule_holdout_readout.py) captures the
final residual **after combining the residual and attention-output paths**.
The frozen teacher's final norm and LM head replay original and edited
predictions to below 1e-6 maximum probability error. It compares rank-k
orthogonal bases from (i) output-weight SVD, (ii) task-blind ordinary-stream
PCA, (iii) labeled edit-delta PCA and (iv) a basis optimized to preserve
teacher causal effects on the fit edits. A full 64-dimensional basis is the
exact local control; zero and random bases are lower controls. A basis is
tested by projecting the edited-minus-original residual, adding that
projection to the original residual, and comparing its output effect with
the native model effect. The reported L1 error sums absolute probability
differences over the vocabulary.

First, the **whole-section featurizer holdout** fits on source-question edits
only and tests on untouched section-key, relevant-color and irrelevant-color
edits. At rank 16, section-key effect L1 error is 1.466 for weight SVD,
.680 for task-blind stream PCA, .470 for source-delta PCA, and .511 for the
source-trained causal basis (full = 0). Relevant-color errors are
1.628/.649/.517/.504 in the same order; irrelevant-color errors are
.354/.199/.184/.183. Fit/dev contain 177/98 source edits, and the section
test has 120 key, 56 relevant-color and 54 irrelevant-color edits. Source
native effects are themselves tiny (L1 .020–.026), so the source-trained
causal objective is a weak signal for discovering a section mechanism.
Source PCA's cross-task transfer is geometric evidence, **not** new-task
discovery: the frozen teacher already trained on section questions.

Second, the **within-section edit-family holdout** fits only section-key
edits (120 fit, 120 dev) and tests on disjoint prefixes with 120 key,
47 relevant-color and 41 irrelevant-color edits. At rank 4, key-edit PCA
has L1 error .301 on key and .567 on held-out relevant-color edits, versus
1.620/1.672 for weight SVD. At rank 16, key-edit PCA has **.002** key error,
**.041** relevant-color error and **.040** irrelevant-color error; full = 0.
The same-rank task-blind PCA has .860/.768/.313. Thus key and color edits
share a compact *local output-response subspace* in this trained teacher.
The 16-dimensional result uses one quarter of the width, and may reflect a
low-dimensional color-output space rather than a small reusable algorithm.

The teacher-effect optimizer selected its PCA initialization at rank 4 and
16, including a lower-learning-rate rank-4 retry. It adds no supported
benefit here. This is a meaningful negative result for that objective, not
evidence that causal supervision is generally useless. The next featurizer
test should seek **context-dependent gates or overlapping groups** within a
faithful shared dictionary and measure sufficiency, necessity and spillover
on held-out key/color combinations. It should also use an independent
teacher seed. Reconstruction, teacher KL, exact-oracle prediction, causal
fidelity and genuinely new-task discovery remain separate endpoints.

Artifacts: [scratch balancing control](results/v8_scope_scratch_balanced_continue_seed101/),
[whole-section source-fit basis](results/v8_scope_rule_holdout_rank16_seed44001.json),
[rank-4 key-fit basis](results/v8_scope_keyfit_colorholdout_rank4_seed46001.json),
and [rank-16 key-fit basis](results/v8_scope_keyfit_colorholdout_rank16_seed46001.json).
