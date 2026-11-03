# Joint task mixture with a cross-section rule

V7 keeps the sampled V14 narrative rules (including lexical/skip competition, brackets, and local conditional actions), scoped records, and score updates. It replaces the old fixed event-report arm with the [variable-clause world renderer](document_world_v5_clause_realizations.py) and adds a fifth, deliberately oversampled `cross_report` kind. One cross report contains an event history, two reviewer-to-ship assignments, and two ship-room-category records in randomized section order. Its final category requires the chain **object + attribute → latest report source → reviewed ship → record category**. A different-key event occurs after the relevant report, so last-name copying is wrong. The answer parser reconstructs the entire chain from visible text only; the sampled world is used to render and to design edits, never to score the model as an oracle. The five document kinds are sampled from one explicit mixture. They remain mostly independent *between* documents; the cross kind supplies a real within-document composition case. This is a structural pilot, not a FineWeb-prevalence estimate.

The [two-panel generator audit](results/document_mixture_v7_cross_audit_n800_seed34701.json) checks 300 cross documents from the training-support split and 300 from a held-out ship-category split per seed, plus 800 mixture documents per seed. All visible-prefix site answers normalize and match, including thousands of narrative rule sites. In each cross panel, reviewer-ship swaps and target-category edits flip the answer, while unrelated report-source edits and section reorderings do not. A relevant report-source edit flips the answer when it remains within the generator's source-versus-actor support: 268/300 and 267/300 cases on the first panel's two splits; unsupported swaps are excluded. The held-out record pair in each composition document is the **queried** ship-category pair, rather than an unrelated record. This fixes a subtle potential split leak.

Form improved on two independent long-document screens, but fails a natural-language match. Each arm contributed 64 first-100-word windows; short record and score documents were excluded, so these are **conditional long-document** comparisons, not full-mixture rates.

| Offset | V6 word types / top-ten share / first-40 repeat | V7 | FineWeb |
|---|---:|---:|---:|
| 30000 | 372 / .313 / .341 | 414 / .314 / .267 | 2451 / .217 / .155 |
| 31000 | 373 / .309 / .341 | 416 / .311 / .264 | 2353 / .218 / .185 |

V7's variable event clauses and cross-section layout reduce repetition, while its vocabulary and dominant grammatical frames remain far too narrow. [Form panel 1](results/document_mixture_v7_vs_v6_fineweb_skip30000_n64.json) and [panel 2](results/document_mixture_v7_vs_v6_fineweb_skip31000_n64.json) save only aggregate statistics and natural-document hashes.

## Behavioral gate after ordinary next-token training

A 64-dimensional, four-head bilinear attention-plus-bilinear-MLP model was trained for 2,000 ordinary next-token CE updates on 10,000 documents at depth 2 and depth 4, one model seed each. A separate calibration stream supplied vocabulary; evaluation text did not. IID and held-out-combination panels each contain 1,000 fresh documents. The composition split withholds specified record ship-category pairs and score start/increment pairs. Exact overlap with training is **40/1000 IID documents** (small record/score support) and **0/1000 composition documents**. Results on the IID panel:

| Blocks | Whole-token NLL | Local conditional-action accuracy | Source-attribution accuracy | Cross answer accuracy | Cross answer on composition panel |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.025 | .939 | .117 | .210 | .242 |
| 4 | .995 | .964 | .123 | .205 | .112 |

The [training record](results/document_mixture_v7_cross_depth2_4_2000.json) also breaks out bracket, record, score and other sites. On 128 new supported local conditional-action edits, the newly correct action's probability rises by **.925** (depth 2) or **.866** (depth 4): that rule passes the behavioral gate ([depth-2](results/document_mixture_v7_depth2_action_causal_n128.json), [depth-4](results/document_mixture_v7_depth4_action_causal_n128.json)).

The cross-section rule fails. On 200 fresh cases per split, full-vocabulary cross-answer accuracy is .255/.245 for the depth-2 model and .220/.140 for depth 4 (training-support/composition split). Changing the relevant earlier report source changes the newly correct category's probability by approximately **zero**: depth 2 +.0001/−.0004 and depth 4 −.0002/+.0004 on the two splits. Reviewer-to-ship swaps also have approximately zero effect. The models do not behaviorally implement the intended chain, so we **must not** search the weights for a cross-source-record circuit in these checkpoints ([depth-2 edits](results/document_mixture_v7_cross_depth2_causal_n200.json), [depth-4 edits](results/document_mixture_v7_cross_depth4_causal_n200.json)). Four blocks have lower aggregate token loss but worse composition accuracy. These are one-seed pilots, not a depth ranking.

## A whole-kind featurizer holdout on the rule that was learned

At the second attention layer of the depth-2 model, I fitted local orthogonal pre-`W_O` bases on **event, record, score, and cross documents only**, with all narrative documents excluded. The model itself saw narrative during training. Two independent fit/test seeds each use 1,500 sampled fit streams, fresh narrative documents, and 128 new supported action edits. Weight-only `W_O` singular vectors use no data; unlabeled PCA uses non-narrative activations; a trace-normalized hybrid and physical/random axes are controls. Rank 64 replays the model exactly. The lower-rank projection is applied at all positions in this one layer.

| Basis / rank | Narrative token NLL, panel 1 / 2 | Conditional-action edit response, panel 1 / 2 |
|---|---:|---:|
| Native / 64 | 1.496 / 1.512 | .897 / .928 |
| Weight only / 16 | 1.562 / 1.581 | .501 / .510 |
| Unlabeled other-kind PCA / 16 | 1.765 / 1.770 | .153 / .131 |
| Hybrid α=1 / 16 | 1.571 / 1.584 | .245 / .219 |
| Weight only / 32 | 1.502 / 1.519 | .837 / .859 |
| Unlabeled other-kind PCA / 32 | 1.558 / 1.566 | .898 / .911 |

This is a useful **task-transfer contrast**, not proof of canonical interpretability. At rank 16, a weight-only local basis retains more of the held-out task effect than unlabeled directions learned from other document kinds; at rank 32 the data basis retains slightly more causal effect but hurts general narrative loss more. Neither basis identifies named semantic variables, and the test is not matched at equal whole-function fidelity. Only one layer and one model seed were studied. The full [first basis panel](results/document_mixture_v7_depth2_action_bases_l2_holdout.json) and [replication](results/document_mixture_v7_depth2_action_bases_l2_holdout_seed35091.json) include all ranks and controls.

The next DGP change should broaden class-appropriate vocabulary and discourse acts, and make more tasks interact inside the same document without simply appending unrelated sections. The next weight-method test should move beyond local bond compression to QK/OV path inspection or a copy-aware functional tensor, and compare causal specificity at matched fidelity. The cross task should be retained as an **unlearned** diagnostic until a model passes its source, assignment, and record edits; do not tune the whole project around forcing this one task to pass.
