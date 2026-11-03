# Game events can ground a broader sports-language world

The earlier sports document DGP could verify final scores, player box-score
facts, and records, but its sentences remained too repetitive. A final box
score cannot by itself support most commentary about the course of a game.
This audit asks whether the same human report games can be joined to an
ordered play-by-play event source before building a broader sampler.

The [audit script](audit_sports_pbp_join.py) reads the 2016/17 season files
from the [NBA Data Archive](https://github.com/jsierrahoopshype/nba-data-archive).
It joins each RotoWire or SBNation report to a game by **exact date, both
complete team names, and ordered home/away final score**. It also checks
quarter scores and roster surnames. The raw archives and human prose remain
local; the [aggregate JSON](results/sports_pbp_report_join_2016.json) stores
counts and SHA-256 hashes of the two input archives. The game-key partition
is the same whole-game train/held-out partition used by the sports DGP.

| 2016/17 report source | In date range | Unique strict joins | Distinct joined games | Quarter scores agree | At least 8 roster surnames agree |
| --- | ---: | ---: | ---: | ---: | ---: |
| RotoWire | 1,593 | 1,593 (100%) | 1,025 | 1,593 | 1,593 |
| SBNation | 1,005 | 864 (86.0%) | 685 | 864 | 864 |

The two sources share 627 distinct joined game IDs. The 1,230-game event
archive has about 598,000 actions, including about 96,000 made shots,
114,000 missed shots, 127,000 rebounds, and 34,000 turnovers. These counts
include every archived game, not just report-matched games. The 141 unmatched
SBNation rows are excluded from an event-grounded sampler; the audit does not
infer their missing games.

An archive peculiarity matters: non-scoring actions often carry a `0/0`
score even after points have been scored. The audit treats `0/0` on such rows
as an absent cumulative score and carries the prior score forward. After
that correction, all joined games have nondecreasing periods and no score
step above four points. One RotoWire-linked game still has a negative
cumulative-score step in the feed; event-derived score trajectories for that
game need inspection or exclusion. Quarter totals nevertheless agree for
every joined report.

The [event-predicate audit](audit_sports_event_predicates.py) independently
checks two candidate game-flow rules. It ignores repeated score snapshots,
requires each positive score step to belong to exactly one side and match
that side's scoring action, and excludes the one invalid trajectory from
future sampling. A **lead change** reverses the last non-tied leader after a
score; a tie alone does not count. An **unanswered run** is consecutive
points by one side within one period, ending when the opponent scores or
the period changes. On distinct matched games, the resulting opportunities
are abundant:

| Source and whole-game partition | Games | Valid score trace | Any lead change | Q4/OT lead change | Run of at least 8 unanswered points |
| --- | ---: | ---: | ---: | ---: | ---: |
| RotoWire train | 810 | 809 | 712 | 276 | 758 |
| RotoWire held out | 215 | 215 | 193 | 70 | 204 |
| SBNation train | 540 | 540 | 474 | 194 | 505 |
| SBNation held out | 145 | 145 | 131 | 45 | 139 |

All matched game traces pass scoring-team and scoring-action attribution.
These are possible facts in the source world, **not** yet evidence that
human reports mention them or that generated sentences express them
correctly. The [aggregate](results/sports_event_predicates_2016.json)
records the exact definitions, counts, and source hashes. Synthetic
tie, period-boundary, and repeated-score edge cases were also checked.

Human reports describe two-sided runs such as `26–5`, not just unanswered
`10–0` stretches. The [run-mention screen](audit_sports_report_run_mentions.py)
looks for numeric `X–Y run` phrases in joined reports and asks whether the
same two scores occur over a contiguous play-by-play interval, allowing
either team orientation. The screen is weaker than full entailment: a
common pair can occur by chance, and it does not certify the named team or
other claims in the sentence.

| Source and split | Clean matched report rows | Rows with numeric run | Run mentions | Match within one period | Match anywhere in game |
| --- | ---: | ---: | ---: | ---: | ---: |
| RotoWire train | 1,252 | 37 | 38 | 27 | 36 |
| RotoWire held out | 340 | 16 | 16 | 9 | 9 |
| SBNation train | 678 | 152 | 212 | 202 | 207 |
| SBNation held out | 186 | 40 | 55 | 51 | 55 |

This is a clear **genre-conditioned act frequency**: numeric runs occur in
roughly 3–5% of RotoWire reports and 22% of SBNation reports. None of these
in-scope human mentions is an unanswered `X–0` run, despite unanswered runs
being common in the source world. Several RotoWire misses are explicitly
cross-period descriptions; other misses need closer alignment. The
[aggregate JSON](results/sports_report_run_mentions_2016.json) records the
scope and source hashes. A sampler should draw event intervals first and
express their verified deltas, while conditioning whether and how often to
discuss runs on genre. It should not copy human run clauses merely because
a number pair matches somewhere in the game.

This is a **source alignment result**, not a realism or entailment pass.
Matching the game does not prove that a sentence about momentum, a foul,
or a late comeback is supported by these events. The separate
["other" sentence screen](results/sports_other_opportunities.json) found
that only about 6–7% of RotoWire and 0.15% of SBNation sentences in its
coarse `other` class have an exact team-stat opportunity under a permissive
nearest-team proxy. Editorial lexical cues occur in about 4% and 21%,
respectively. These overlapping, noisy screens motivate broader event and
discourse state; they are not semantic annotations.

The next DGP should sample a latent game trajectory with event-derived
facts such as lead changes, scoring runs, and period transitions. It should
then sample document acts and sentence realizations, with every marked
answer recoverable from visible text and event state. First validate the
event predicates independently against the feed and source-game splits;
then compare full-document and rule-window form against held-out human
reports. Only train bilinear models on the broader DGP if that form screen
improves, and check which sampled rules models actually learn before
interpreting their weights. The downstream purpose remains testing
weight-based canonicalization, sparsification, and causal component reuse
against known and behaviorally learned rules.

Reproduce locally, with the two uncommitted raw archives available:

```bash
python language_task_suite/audit_sports_pbp_join.py \
  --pbp /path/to/nbastatsv3_2016.tar.xz \
  --shotdetail /path/to/shotdetail_2016.tar.xz \
  --output language_task_suite/results/sports_pbp_report_join_2016.json

python language_task_suite/audit_sports_event_predicates.py \
  --pbp /path/to/nbastatsv3_2016.tar.xz \
  --shotdetail /path/to/shotdetail_2016.tar.xz \
  --output language_task_suite/results/sports_event_predicates_2016.json

python language_task_suite/audit_sports_report_run_mentions.py \
  --pbp /path/to/nbastatsv3_2016.tar.xz \
  --shotdetail /path/to/shotdetail_2016.tar.xz \
  --output language_task_suite/results/sports_report_run_mentions_2016.json
```
