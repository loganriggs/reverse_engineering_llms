# Event-grounded run arithmetic works; the first prose renderer fails form

The [latent world sampler](sports_event_world_dgp.py) supplies a real
2016/17 game trajectory, its final and period-end scores, top box-score
scorer, and a chosen within-quarter scoring interval. The
[first renderer](sports_event_world_renderer.py) samples a result sentence,
player fact, optional period facts, and (at a genre-dependent rate) a
three-sentence run block. Two score snapshots appear before the run claim,
so the two `X–Y` numbers are computable from visible text. Result sentences
are drawn from the existing guarded RotoWire/SBNation score-form bank when
possible; a plain, unambiguous fallback is used otherwise. Borrowed
weekday/time words are excluded from the bank because they need not match
the sampled game. Overtime scores stay in the latent world, while this
pilot's run sentences select regulation quarters only.

The [audit](audit_sports_event_world_renderer.py) forced a run in 100
sampled documents for each source × whole-game partition. It checked the
visible-prefix arithmetic at **800 numeric run-answer sites**, all result
scores/winners, 400 player-point facts, and 684 period-score and leader
claims. Those checks pass. A tie, period-boundary, and score-snapshot audit
also supports the underlying event extraction. The chosen game remains
held out as a unit across the two human sources.

But the language-form screen decisively fails. On 64 generated and 64
human first-30-word crops per arm, the share of trigram occurrences also
seen in another document is:

| Source / game split | Generated pilot | Human reports |
| --- | ---: | ---: |
| RotoWire train | .672 | .217 |
| RotoWire held out | .694 | .246 |
| SBNation train | .651 | .044 |
| SBNation held out | .649 | .052 |

The first-40-word screen gives the same conclusion: generated
.665/.703/.682/.723 versus human .170/.207/.064/.060 in the same arm order.
The 40-word screen is additionally **selection biased**: short generated
documents are rejected, which overselects the long run block. For example,
17/64 selected RotoWire train documents contain a run although the default
run insertion rate is 4%; the 30-word panel selects only 1/64. The
[30-word aggregate](results/sports_event_world_renderer_n100_form64_w30.json)
and [40-word aggregate](results/sports_event_world_renderer_n100_form64.json)
record draw attempts, selected run counts, and additional form measures.
Only generated examples and natural aggregate metrics are saved.

The generated prose repeats score/quarter scaffolding and often reads as a
series of isolated facts. That is not close enough to the human reports to
make this the main training DGP. **No model was trained on this pilot**, and
the successful arithmetic check is not evidence that a bilinear model has
learned the rule or that a weight decomposition can recover it.

Decision: keep this as a causal arithmetic and visibility fixture, but stop
expanding the fixed sentence bank as a realism strategy. The next renderer
needs richer discourse planning and clause combination, with independent
human-surface comparisons at matched document positions. A natural-carrier
arm with event-verified insertions was subsequently tested as a
[form control](SPORTS_EVENT_NATURAL_CARRIER_RESULT.md): natural background
helps, but its local run windows still repeat far more than human ones.
After a form improvement, train multiple small bilinear models,
confirm learned run/score/other rules behaviorally on whole-game and
composition holdouts, and only then compare weight canonicalization and
sparsification at matched task fidelity.

Run locally with the season archives available at the paths specified in
`sports_event_world_dgp.py`:

```bash
python language_task_suite/audit_sports_event_world_renderer.py \
  --semantic-n 100 --form-n 64 --words 30 \
  --output language_task_suite/results/sports_event_world_renderer_n100_form64_w30.json
```
