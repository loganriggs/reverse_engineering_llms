# Answer-free intervention scan across later attention heads

## Test

We reused the two disjoint 800-edit panels from
[the intervention-gauge experiment](LATER_HEAD_UNLABELED_INTERVENTION_RESULTS.md).
Each panel contains 80 one-token edits of each of ten types. We captured
the **native 128-dimensional final-position output of every head** in
layers 9, 11, 13, and 15 of the 546M bilinear-attention model. No task
answer, downstream gradient, or edit-type label was used to rank heads or
fit axes. For each of 36 heads, we fitted the leading uncentered PCA axis
to its 800 training edit responses and ranked heads by that axis's raw
response energy. The independent validation panel supplied edit types
only for interpretation. We compared the top-decile edit cluster from
the axis with the cluster from the complete 128-dimensional head-response
norm.

| Unlabeled energy rank | Layer/head | Train energy | Validation/train energy | Leading validation edit type: axis / full-head norm (of 80) |
|---:|---:|---:|---:|---|
| 1 | 13/8 | 4,999 | .98 | bracket 80 / 80 |
| 2 | 13/3 | 1,236 | .90 | punctuation 70 / 79 |
| 3 | 11/5 | 893 | 1.00 | bracket 80 / 79 |
| 4 | 11/6 | 614 | .91 | bracket 69 / 43 |
| 5 | 11/3 | 518 | .93 | noun-number 78 / 69 |
| 6 | 13/5 | 347 | .69 | punctuation 77 / 56 |

The top-ranked head is the already known bracket path. The scan also
surfaced punctuation-sensitive heads 13/3 and 13/5 without using
punctuation labels to choose them. Head 11/5 appears to be another
bracket-responsive candidate, while 11/3 is the previously known
agreement candidate. A single coordinate improves input-edit clustering
over the head norm for 11/6, 11/3, and 13/5; it does not improve 13/8
or 13/3. Energy order alone is not a rule-discovery score: high-energy
heads can respond to surface changes without helping the corresponding
model behavior.

These comparisons are **native head response**, before any rank-four
reduction, so this test addresses head selection and candidate input
variables. It does not yet establish which heads causally affect answers,
whether apparently related heads duplicate computation, or whether the
punctuation cluster predicts an independent output behavior. The same
ten edit types occur in both panels; this is not a whole-task discovery
holdout. Next, choose candidate output predictions from one panel, freeze
them, and test finite patches on disjoint token forms, contexts, and
task families. Match patch dose and include full-head-norm-selected
heads and random head/axis controls.

Reproduce with `flagship_all_head_unlabeled_edit_scan.py`. Per-head
response arrays and scores are in `full_width_results/all_head_unlabeled_edit_scan.npz`
and `.json`.
