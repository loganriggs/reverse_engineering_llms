# Task-rich document mixture: scope, score, events, and local rules

The [V5 generator](document_mixture_v5.py) adds two contrastive task families
motivated by the [natural prediction-site red team](FINEWEB_PREFIX_SENSITIVITY.md)
to the existing sampled language. A draw first chooses one of four document
kinds: the V14 narrative stream, an event report, a two-record listing, or a
score report. This is an explicitly **task-rich pilot mixture**, not an
estimate of FineWeb's document-kind or dependency frequencies.

The narrative arm retains V14's joint bigram-like lexical, previous-place,
induction, lead-cue, and bracket competition, together with its later fact
families. Its scored sites use the exact V14 visible-prefix next-token
distribution. The event-report arm uses one visible state history for aliases,
attributed updates, and later owner/color/place recaps. The new record arm
uses two ships with the *same room number* and different categories; the
query names one ship. The score arm gives an initial two-team score and a
scoring event, then states an updated total. Half its documents also print
the final score early as a deliberate answer-leak control.

For example, a generated record can say that Aurora's room 6085 has category
11B and Horizon's room 6085 has category 9C, then ask for Aurora's category.
Swapping only the named ship changes the correct answer. Reordering the two
records does not. A generated score report can start with 2 points and add
3, giving 5; editing the scoring event changes the answer, while toggling
an early headline between present and absent does not change the oracle.
They are sampled under one document-kind mixture, although each kind still
uses a separate, stylized surface grammar.

The [800-document audit](results/document_mixture_v5_audit_n800.json)
replayed 200 documents per kind and checked **4,702 scored sites** against
their visible-prefix distributions or independent semantic parsers. It
passed 200/200 scope-key edits, 200/200 record-order nuisance edits,
200/200 score-event edits, and 200/200 headline-leak toggles. Among the
record documents, 104 queries name the last listed ship and 96 name the
other ship, so a recency heuristic is an explicit evaluation subgroup.
Exactly 100 score documents leak the answer in a headline and 100 do not.
The audit also counts each local and semantic site family; some V14 families
remain rare in 200 narrative documents, so larger training/evaluation
draws need minimum per-family coverage.

The natural-form gate still fails. In 49 eligible documents × 100 alphabetic
tokens, V5 has **355 word types, 32 singletons, and top-ten share 30.7%**.
A same-size V14 diagnostic control has 327 types, 16 singletons, and 36.4%
top-ten share. The independent FineWeb preview screen has 1,993 types,
1,336 singletons, and 21.45% top-ten share. The V5 comparison excludes
short record and score documents that have fewer than 100 alphabetic tokens;
its eligible set contains 27 narratives and 22 event reports. Thus the
modest improvement comes from mixing two long-form grammars, while the
overall language remains narrow and repetitive. This is a structural and
task-coverage improvement, **not** a realism pass.

The next training experiment can use this mixture as a controlled pilot:
train 1–4-layer bilinear models on next-token loss, report each document
kind and rule separately, and test scope-key sensitivity, record-order
invariance, score-event sensitivity, leaked/unleaked accuracy, and V14
local-rule transfer. Keep entire task families out of featurizer training
for a later discovery test. Before promoting V5 as the main language
benchmark, expand document kinds, discourse plans, and class-appropriate
lexicons using a separate FineWeb development panel, then evaluate natural
form on untouched documents. V5 has no exact full next-token oracle for
the semantic arms; their **scored** sites do have independent visible-prefix
oracles.

## Bilinear full-language-model pilot

The [trainer](train_document_mixture_v5.py) fit one seed each of 1–4
bilinear-attention-plus-bilinear-MLP blocks, width 64, four heads, RMSNorm,
on 20,000 independently sampled V5 documents for 2,000 ordinary next-token
CE updates. A separate 2,000-document draw gave the following token losses.
The [full result](results/document_mixture_v5_bilinear_depth1_4_2000.json)
also breaks out every document kind and scored family.

| Blocks | Held-out token NLL | Unleaked score accuracy, fresh 200 | Score-event edit TV | Scope-key edit TV | Record-order edit TV |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.071 | .600 | .507 | .039 | .087 |
| 2 | 1.013 | 1.000 | .908 | .002 | .007 |
| 3 | 1.001 | .340 | .732 | .019 | .131 |
| 4 | .998 | 1.000 | .947 | .016 | .187 |

The [fresh paired panels](eval_document_mixture_v5_causal.py) use
candidate-normalized distributions; the oracle TV is **1** for a changed
score event or ship key, and **0** for a record-order change. Depths 2 and 4
learn the score mapping *within the training range* even without a headline:
they correctly answer 200/200 fresh examples and respond strongly when the
scoring event changes. The three-block run has lower overall token loss than
the two-block run yet much worse score accuracy, so aggregate language loss
does not rank this task's competence. All four runs remain nearly blind to
the ship key. The four-block model scores 65.4% when the queried ship is the
last record but only 23.7% when it is not; reordering records changes its
category distribution much more than changing the query key. This is a
position shortcut, not recovered scoped retrieval.

The score competence is also **narrow**. On the [transfer panel](results/document_mixture_v5_depth2_2000_causal_transfer_n200.json),
the two-block model gets 0/156 examples with a four-point increment, which
was absent from training. It gets 12% when asked for the other team's score,
which was never a training query; changing the first team's scoring event
wrongly moves its answer distribution by TV .906, versus the oracle's 0.
The [four-block transfer panel](results/document_mixture_v5_depth4_2000_causal_transfer_n200.json)
has the same 0/156 and 12% failures, with irrelevant-event TV .943. Thus
these runs learned a reliable *in-distribution left-team update*, not a
general addition rule or a complete scoreboard representation. One seed
per depth and 2,000 steps cannot support an architectural ranking.

The independent test draw has substantial exact reuse for the two small
document grammars: the [overlap audit](results/document_mixture_v5_exact_overlap_20000_2000.json)
finds **135/283 score** and **163/399 record** test documents identical to
some training document. It reproduces the trainer's kind counts while
skipping expensive V14 text generation, and a separate full-generation hash
check gave the same counts. The fresh edit and out-of-support probes are
therefore more informative than the IID accuracies, but they do not turn
this into a clean compositional generalization result. The next training
split should reserve entire initial-score × increment combinations and
record field assignments before drawing examples.

The next version should preserve the whole task mix while varying the queried
team and increment, reserving unseen initial-score × increment combinations
for evaluation. It should keep the leak toggle and record-order controls.
Only after a model passes those transfer edits is the score subcircuit a
stronger target for comparing tensor reduction, sparse bases, and causal
featurizers. The current successful local conditional-action task remains
another, different computation to include in that comparison.
