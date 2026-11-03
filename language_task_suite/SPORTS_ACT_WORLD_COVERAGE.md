# The sports world is missing much of the report

The current sampled sports document always emits one result, five player-stat
clauses, and three explicit answer recaps. A previous quarter variant added
six fixed period acts and made phrase repetition worse. Before implementing
another document planner, this screen asks how much of the *human* first twelve
sentences the current latent world could even support.

The [audit](audit_sports_act_world_coverage.py) uses the existing transparent
lexical act proxies over the official RotoWire and SBNation training archives,
partitioned by game key. It marks result, player-stat, and game-flow proxies
as potentially supported by final-score, box-score, and quarter facts. It
counts background, outlook, and other primary proxies as outside the current
world. These are **opportunity estimates**, not certified discourse labels or
proof that a generated clause would be natural.

| Source / game partition | Reports | Primary acts outside current world | Reports with a visible W–L pair matching box score |
| --- | ---: | ---: | ---: |
| RotoWire train | 2,690 | 46.13% | 73.49% |
| RotoWire held out | 708 | 46.32% | 72.88% |
| SBNation train | 6,065 | 63.83% | 7.58% |
| SBNation held out | 1,568 | 64.02% | 8.29% |

Every RotoWire row and all but two SBNation rows have parseable team wins and
losses. Among RotoWire training reports, 1,326 sentences with a matching
win–loss pair are also labeled as result sentences. That suggests a useful
*RotoWire-specific* next world variable: prior team records, a sampled game
outcome, and the resulting records. It would allow a result lead to include a
true standings update, rather than adding a filler sentence. An independent
visible-text parser must verify the record and selective edits must update it
when the winner changes. A numeric pair alone is not semantically certified:
it might describe a pregame record, a series, or something else.

This result also rules out treating RotoWire and SBNation as mere surface
styles of the present world. SBNation has far more long-form game flow and
unclassified material; its visible record matches are rare. The next DGP
should first build a style-conditioned **act inventory and evidence selector**
that can omit, combine, or revisit facts, rather than permuting nine mandatory
sentences. Adding record updates is one grounded RotoWire arm; broader
SBNation-like prose needs additional world state and independently verified
act classes. Audit full documents and scored windows, then test behavioral
learning of each rule before comparing weight-based interpretability methods.

Aggregate counts are in `results/sports_act_world_coverage.json`; no natural
report text is saved. The earlier corpus labels are lexical proxies, so the
46%/64% figures should guide design, not serve as a strict mathematical lower
bound on all possible generated prose.
