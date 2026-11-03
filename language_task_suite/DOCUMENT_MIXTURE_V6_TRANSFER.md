# Whole-combination transfer in the task-rich document mixture

V5 showed that a bilinear model could answer trained-range score updates
while failing both a new increment and a question about the unchanged team.
It also left a large fraction of short IID documents identical to training
documents. [V6](document_mixture_v6.py) keeps the same four document kinds
and V14 local-rule arm, but samples score increments 1–4 and queries either
team. The event always changes the first team's score; a correct answer for
the second team must ignore it.

The score composition split reserves initial-first-team score × increment
pairs by a fixed modular fold. For records, it reserves ship × category
assignments and excludes every held-out assignment from training records;
composition queries name a held-out assignment. These are **whole
combinations**, not random document rows. The [sampler audit](results/document_mixture_v6_audit_n1600.json)
replayed 200 documents per kind in each split. It checked every scored
visible-prefix distribution, 400 record-key edits, 400 record-order
controls, 400 score-event edits with the expected changed or unchanged
answer, and 400 headline toggles. No record or score document in the
composition audit duplicated a training document. Score questions and
record recency conditions are close to balanced in each split.

The natural-form gate remains negative: 49 eligible V6 documents × 100
alphabetic tokens contain 352 word types, 46 singletons, and a 29.8%
top-ten share. The earlier independent FineWeb preview screen had 1,993
types, 1,336 singletons, and a 21.45% top-ten share. This revision tests
transfer, not realistic language form. Only the V14 narrative arm has a
full next-token oracle; the semantic arms have independent oracles at their
scored prediction sites.

## Trained-model result

One seed of the original width-64, four-head bilinear transformer was
trained for 2,000 full next-token CE updates at each depth 1–4 on 20,000
sampled documents. The [training record and checkpoints](results/document_mixture_v6_bilinear_depth1_4_2000.json)
include 1,000 IID and 1,000 held-out-combination documents per model.
The score and record arms have small enough combinatorial domains that the
IID test still repeats 45/152 score and 112/202 record documents exactly;
the composition test repeats zero. Aggregate token NLL is nearly the same
on IID and composition (depth 4: .969/.970), despite large task failures.

Fresh paired candidate-normalized panels with 200 records and 200 score
reports per split were evaluated at each depth. The [depth-4 result](results/document_mixture_v6_depth4_causal_n200.json)
is the clearest local learner: IID scoped-record accuracy is 68% when the
queried record is nonlast and 67% when last, but only 4.5%/5.4% for the
held-out ship×category assignments. Its response to changing the queried
ship is substantial in composition (candidate-distribution TV .48), yet
usually points to the wrong category. Thus sensitivity to a key edit is
not evidence of correct scope binding or transfer.

Score queries distinguish the changed first team from the unchanged second
team. At depth 4, IID accuracy is 39.5% for the first team and 52.6% for
the second; held-out-combination accuracy is 7.8%/63.6%. Event edits change
the model's candidate distribution even for the *unchanged* team (TV .28
in composition), where the exact oracle effect is zero. The model has not
learned a reliable two-team state update. Depths 1–3 likewise fail the
record composition split (all 0%) and have weak first-team score accuracy.
The detailed paired results for [depth 1](results/document_mixture_v6_depth1_causal_n200.json),
[depth 2](results/document_mixture_v6_depth2_causal_n200.json), and
[depth 3](results/document_mixture_v6_depth3_causal_n200.json) preserve
the split and edit breakdowns.

There is a competent control within the same mixture: deterministic V14
`conditional_action` full-vocabulary accuracy is 97.8–100% at depths 2–4
on IID and 98.9–99.6% on the composition draw. That licenses a bounded
local-action circuit study, not a claim that the semantic binding tasks
or natural language were learned. We stop scaling the isolated score/record
task here. The next DGP work should broaden ordinary document and lexical
forms based on real prediction sites while retaining exact sampled oracles,
whole-combination holdouts, and causal no-effect controls. After a model
passes a particular task gate, compare interpretability methods on that
task and keep genuine new-task discovery separate from in-task fidelity.
