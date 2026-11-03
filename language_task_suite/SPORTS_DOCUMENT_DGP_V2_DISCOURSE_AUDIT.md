# Sports-document V2: discourse movement helps rule windows, not whole reports

This experiment tests whether moving known rule-bearing clauses through a
sampled report can close the V1 language-form gap. It does not optimize model
capability. The same three visible-prefix rules remain winner, final-score
margin, and leading scorer among the five listed players.

V2 adds two independent DGP controls to
[`sports_document_dgp.py`](sports_document_dgp.py): `lead_stat_rate` can put
one player clause before the result, and `discourse_entropy` can place the
winner and margin recaps after different stat clauses. The top-player recap
remains after all five player facts, so its answer is visible. The score
parser now locates the result sentence within the visible prefix. Sampling
also partitions source games by a stable hash of date and the two teams;
all appearances of one key across both corpora receive the same partition.
Of the source training rows with unambiguous team cities, 2,689 RotoWire and
6,044 SBNation rows fall in the new training partition; 707 and 1,564 fall
in heldout. Templates still come from the common source bank, so this is a
**game holdout, not a template holdout**.

The [full semantic/form audit](audit_sports_document_dgp.py) ran 300 plans
in each of eight source-style × form-entropy × paired-clause settings,
separately for the training and held-out game partitions. All **4,800**
plans passed their three visible-prefix answers and all **14,400**
team-role, score, and player-point selective edits. The trainer now samples
these disjoint game partitions and constructs its vocabulary from the source
lexicon rather than held-out sampled documents; a one-step depth-1 GPU smoke
test passed. No substantive V2 model training or weight analysis has been
run.

The decisive form comparison realizes **the same plans** in V1 and V2, with
the same source style, player facts, and high surface-entropy settings. On
256 first-55-word documents per partition:

| Partition | V1 whole-document repeated trigrams | V2 whole-document repeated trigrams | V1 winner/margin/top windows | V2 winner/margin/top windows |
| --- | ---: | ---: | --- | --- |
| Training games | 68.7% | 69.9% | 71.4 / 71.7 / 70.5% | 64.7 / 63.4 / 62.7% |
| Held-out games | 69.2% | 70.8% | 71.0 / 71.8 / 71.6% | 64.0 / 64.5 / 64.3% |

The 64-document paired check has the same direction: whole-document share
51.5→53.4% on training games and 53.0→54.9% on held-out games. Target-site
windows improve about 7–12 points. Repeated-trigram share depends strongly
on panel size, so compare only equal-size rows; the 256-document percentages
must not be compared directly to V1's published 64-document natural control.
On the separate 64-document V2 audit panels, the best RotoWire-style arms
still repeat around 54–55% of trigrams, versus 16.0% in the natural RotoWire
control and 5.4% in the natural SBNation control. These natural panels are
genre-matched but not content-matched.

The [corpus player-sentence screen](audit_player_stat_form_ceiling.py)
explains why moving clauses alone has a ceiling. Of 9,113 RotoWire and
5,520 SBNation sentences with exactly one box-score-matched player/point
fact, 5,475 and 4,285 also contain another numeric token. Under a deliberately
**narrow** points-only sentence pattern (nine simple verbs, at most a short
outcome/time qualifier), only 5 and 12 sentences match. That strict count is
not the number of all semantically safe natural sentences; it measures how
rare our current simple prose pattern is in these corpora. Natural reporting
often combines points with rebounds, assists, game trajectory, comparisons,
and other contextual claims that the V1/V2 plan cannot yet support.

**Decision:** preserve V2 as a controlled discourse-order intervention and
the grouped-split training scaffold. Do not present it as a realistic language
distribution or spend a large model/weight-method sweep on it. The next
generator revision should sample more of the underlying world—additional
player statistics, supported game events and comparisons, and what a report
chooses to mention—then realize those facts in varied typed clauses. Audit
factual validity, actual samples, task-window and whole-document form, and
grouped held-out games before using behaviorally competent models to compare
weight-only, sparse, and full-path interpretability methods. Any proposed
new rule must be checked from the visible prefix and a selective edit.

Reproduce the paired form result with
`python language_task_suite/compare_sports_discourse_form.py --n 256 --output language_task_suite/results/sports_document_dgp_v2_paired_form_n256.json`.
The full train/heldout audits and player-form screen are in the corresponding
scripts and aggregate JSON under [`results`](results/). Raw source reports
remain local outside Git.
