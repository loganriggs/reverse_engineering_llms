# Natural surrounding prose does not fix the run-site form gap

The first event-world renderer made correct, visible run-arithmetic targets
but highly repetitive documents. This control inserts the same three
event-verified run sentences into **human reports about the same games**.
It preserves all human source sentences and changes only the insertion.
The [carrier code](sports_event_natural_carrier.py) uses a unique strict
game/event join, a matching final-score pair visible somewhere in the
human report, at least 100 source tokens, and the existing whole-game
train/held-out split. The score need not be in the first sentence. That
common filter yields 622/169 eligible RotoWire train/held-out rows and
450/129 SBNation rows. This is an empirical-carrier form control, not the
fully sampled DGP.

The [audit](audit_sports_event_natural_carrier.py) selects 64 distinct
games per source × partition. For each game it compares the same report's
first 100 words before and after the event block is inserted. The share
of trigram occurrences repeated across documents is:

| Source / game split | Original human report | Same report + run block |
| --- | ---: | ---: |
| RotoWire train | .216 | .346 |
| RotoWire held out | .250 | .403 |
| SBNation train | .114 | .293 |
| SBNation held out | .107 | .302 |

The local task window remains much worse. Ten resamples of ten windows,
each with up to 25 words before and 16 after the numeric run target, give
mean repeated-trigram shares:

| Source / split | Inserted run window | Natural human run window |
| --- | ---: | ---: |
| RotoWire train | .330 | .089 |
| RotoWire held out | .320 | .139 |
| SBNation train | .317 | .049 |
| SBNation held out | .334 | .070 |

The natural-run comparison pools are small for RotoWire (26 train games,
11 held-out games), and ten-window resamples are noisy; the
[aggregate JSON](results/sports_event_natural_carrier_n64_w100.json)
records every resample and pool size. The first-100-word panels are paired
by game, while the local generated and natural-run windows are not paired.
The visible-prefix parser verified every generated run answer encountered
in the panel (132/132/168/164 numeric sites by arm). No raw human passages
are saved in the aggregate.

**Interpretation:** natural surrounding prose helps the document-level
metric, but the inserted task explanation is itself too formulaic.
Merely embedding a fixed three-sentence arithmetic block in realistic text
does not meet the realism objective. Also, matching the event game and its
run numbers does not certify that every human sentence is caused by the
same event facts, so this remains a form control rather than a fully
symbolic language with a complete oracle.

Stop this block-insertion variant. The next informative experiment is to
express event and score changes within more diverse clauses and discourse
relations, then audit **both** task windows and whole documents at matched
positions. Keep whole-game holdouts. Only after the generator passes that
screen should we train the main bilinear comparison and check behavioral
rule acquisition before interpreting weights.

Reproduce locally with the raw archives available at the configured paths:

```bash
python language_task_suite/audit_sports_event_natural_carrier.py \
  --n 64 --words 100 \
  --output language_task_suite/results/sports_event_natural_carrier_n64_w100.json
```
