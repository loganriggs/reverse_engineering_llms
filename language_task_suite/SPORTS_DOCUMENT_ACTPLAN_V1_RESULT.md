# Variable sports act plan: correct rules, insufficient language form

This pilot tests whether sampling the *inventory* of evidence and answer acts
helps more than adding player-stat paraphrases. The [generator](sports_document_actplan_v1.py)
retains guarded corpus-derived result forms and the fact-matched empirical
player-stat bank. It samples one to three rule-answer sites, two to five
player facts (all five when a top-player comparison is asked), a style-biased
result position, and optional pregame and postgame team-record statements.
Pregame win–loss priors are drawn from source-game partitions; a sampled win
or loss updates them exactly. These are synthetic record states, **not**
transplanted factual claims about the source game.

On 100 documents per RotoWire/SBNation × train/held-out game partition, the
[semantic audit](audit_sports_document_actplan_v1.py) checked every marked
answer against visible text, every emitted record against the sampled world,
and its +1 win/loss update. It rerendered all 400 worlds under team swaps and
score edits, all top-player-bearing worlds under point edits, and all
record-bearing worlds under a record-prior edit. All supported checks passed.
The rule inventory genuinely varied: each of winner, margin, and top-player
sites occurred in only about 29–57 of 100 documents per arm. RotoWire led
with the result in 74/82 of 100 documents across the two partitions; SBNation
did so in 11/16, broadly consistent with the earlier human lead-position
proxy. This validates generator semantics and act-selection variability, not
model learning or linguistic realism.

For 64 jointly eligible first-55-word crops, the full candidate is still far
from human sports prose. Repeated-cross-document trigram share is:

| Source / game partition | Fixed empirical stats | Variable plan with default records | Human reports |
| --- | ---: | ---: | ---: |
| RotoWire train | .551 | .556 | .180 |
| RotoWire held out | .545 | .562 | .163 |
| SBNation train | .601 | .623 | .070 |
| SBNation held out | .582 | .614 | .051 |

The scored windows also give no consistent improvement: winner windows
improve on one RotoWire panel but worsen on the other; margin windows worsen
in all four panels. A separate matched-world [record ablation](sweep_sports_actplan_record_ablation.py)
isolates the cause. With no record acts, repeated-trigram share is
.483/.504 on RotoWire and .571/.572 on SBNation train/held-out panels.
Forcing pregame records raises it to .557/.569 and .626/.631; forcing both
pre- and postgame records raises it to .560/.588 and .639/.657. The fixed
empirical-stat baseline on these same jointly eligible worlds is
.536/.544 and .603/.596. Thus variable act selection itself can reduce
repetition, while the tiny record grammar more than erases that benefit.
Top-ten word share does not improve consistently in the record ablation.

Generated examples remain list-like: a result or player-stat lead, several
separate stat clauses, a record statement, then an explicit answer recap.
Although the record rule is true, its current forms are often awkwardly
positioned and repetitive. An earlier audit showed W–L records are a real
RotoWire opportunity, but merely adding two record templates does not
reproduce how human writers integrate that information into a result lead.

**Decision:** retain V1 as a truth-checked act-selection and record-update
fixture. Do not train it as the main improved-language DGP or interpret its
weights as a realism result. The no-record act-selection arm is a useful
controlled DGP hyperparameter; the next substantial version needs a broader
semantic world and corpus-grounded **combined** result/record and game-flow
realizations, plus style-conditioned act selection. Full-document and
task-window form screens must pass much stronger checks before the main model
and weight-method comparison. Model behavior remains the gate for every rule.

Aggregates are in `results/sports_document_actplan_v1_n100_form64.json` and
`results/sports_actplan_record_ablation_n64.json`. They include generated
examples but no natural source passages. The generator provides selected-site
visible-answer oracles, not an exact full-prefix next-token oracle; any model
training uses ordinary full next-token cross-entropy.
