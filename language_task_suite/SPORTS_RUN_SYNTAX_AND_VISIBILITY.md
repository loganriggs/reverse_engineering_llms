# Human run prose is integrated; its numbers are usually latent game facts

The event-world text pilot and natural-carrier insertion both fail at the
local run window. To understand why, the [syntax audit](audit_sports_run_syntax.py)
screens numeric `X–Y run` mentions in 2016/17 reports with a strict event
join, visible matching final score, at least 100 source tokens, and the
same whole-game train/held-out partition. It records only aggregate
surface statistics, not source sentences or semantic labels.

| Source / game split | Run mentions | Mean sentence tokens | Sentences ≤16 tokens | Comma or semicolon | Another number in sentence | At least 2 score pairs in prior 3 sentences or current prefix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RotoWire train | 24 | 32.4 | 0 | 17 | 11 | 7 |
| RotoWire held out | 11 | 34.6 | 0 | 11 | 5 | 4 |
| SBNation train | 165 | 34.1 | 5 | 113 | 95 | 52 |
| SBNation held out | 47 | 32.0 | 1 | 30 | 23 | 10 |

The run appears around 40–50% of the way through the sentence on average.
Full team names appear in **none** of these screened run sentences; team
nicknames appear in many, while other sentences use cities or references.
These facts contrast with our short, three-sentence inserted block, which
repeated full team names and devoted an entire sentence to each snapshot.
Lexical clause cues are common, but the cue screen is broad and does not
prove a distinct clause or discourse relation. The
[aggregate](results/sports_run_syntax_2016.json) includes exact cue counts
and common two-word contexts around `run`.

There is a deeper task-design issue. Only 7/24 and 4/11 RotoWire run
mentions, and 52/165 and 10/47 SBNation mentions, have **even two**
score-shaped pairs in the preceding three sentences or earlier part of the
current sentence. That is an upper-bound opportunity screen: those pairs
can be final scores, unrelated quarter scores, or earlier run claims, not
necessarily the two endpoints needed to calculate `X–Y`. Thus exact
run-number prediction from fully visible scoreboard arithmetic is unusual
in these retrospective reports. Requiring two explicit snapshots before
each run makes the arithmetic learnable but changes the genre.

**Course correction:** keep the verified event world and run facts as a
latent narration mechanism, but stop treating numeric run completion in a
retrospective article as a primary naturalistic one-token task. A distinct
live-report or score-ledger genre can legitimately make score transitions
visible and test arithmetic. For retrospective prose, the visible tasks
should instead emphasize what reports actually expose—winner and margin
from the printed result, entity/reference continuity, syntax, source
attribution, and document style—while run wording is sampled from the
latent game state and evaluated for factual correctness and form, not
assumed inferable from the immediate prefix. This expands task diversity
and avoids a false mechanistic target.

The next experiment should test a **mixture** of these two genres with
shared teams and event rules but different observation regimes. First
measure whether score-ledger contexts make run arithmetic genuinely
predictive, then check whether retrospective prose can approach human
task-window variation through clause-level realization. Train and analyze
bilinear models only on rules they behaviorally acquire under whole-game
and held-out-combination tests.

Reproduce locally:

```bash
python language_task_suite/audit_sports_run_syntax.py \
  --output language_task_suite/results/sports_run_syntax_2016.json
```
