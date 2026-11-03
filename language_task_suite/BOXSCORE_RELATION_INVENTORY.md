# Which natural player-stat facts can the box score support?

The [strict player-sentence screen](GROUNDED_PLAYER_SENTENCE_BANK_YIELD.md) left only 13 RotoWire and 33 SBNation training clauses under a points/rebounds/assists-only world. This [inventory](audit_boxscore_relation_inventory.py) asks whether adding existing box-score fields could materially enlarge the candidate pool. It is an **opportunity screen**, not a safe sentence bank or a semantic oracle.

For each source sentence with exactly one full roster-player mention, the audit recognizes digit or spelled-out-zero-to-twenty quantities adjacent to points, rebounds, assists, steals, blocks, turnovers, or made threes. It checks each quantity against that player's PTS, REB, AST, STL, BLK, TO, or FG3M field and separately checks whether every numeric token in the sentence was accounted for. It groups source games by the stable train/held-out partition and saves only aggregate counts.

| Corpus/group | One-player sentences with recognized stat | All recognized values match | Every number accounted for | Of those, need extended field | Games with numeric candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| RotoWire train | 8,105 | 6,882 | 3,069 | 772 | 1,401 |
| RotoWire held-out | 2,089 | 1,752 | 810 | 208 | 367 |
| SBNation train | 7,784 | 4,127 | 1,511 | 338 | 1,147 |
| SBNation held-out | 2,010 | 1,106 | 418 | 90 | 319 |

In the 772 RotoWire training candidates newly covered by extended fields, **steals appear in 534**, **blocks in 300**, turnovers in 18, and made threes in 36; fields can co-occur. The corresponding SBNation counts are 168 steals, 128 blocks, 81 turnovers, and 37 made threes across 338 candidates. Thus adding steals and blocks is a plausible *world-variable* expansion, with a smaller but nonzero made-threes opportunity.

The difference between **3,069 numeric candidates and 13 strict RotoWire clauses** is the central result. Among the 3,069, lexical markers for history, game flow, and outcome occur in 207, 655, and 245 sentences respectively, with overlap. Manual review found broader unsupported claims too: “team-high,” “led all scorers,” “off the bench,” “career-high,” qualitative evaluations, other players mentioned by surname or pronoun, and shooting or injury context. Some of these can be checked with richer box-score or game-history state; others cannot. Matching numbers is therefore **necessary for slot substitution but nowhere near sufficient** for semantic validity. Conversely, the strict screen is intentionally low-recall and rejects many valid sentences; its small yield is not a population estimate of all safely usable natural forms.

The parser is deliberately limited: it does not handle written numbers above twenty, complex shooting fractions, abbreviated stats without a number-unit phrase, or every way to refer to another player. Its one-player test can miss surname-only references. Adversarial fixtures checked matching spelled-out steals, an extra unbound year, and a mismatched stat. The [aggregate](results/boxscore_relation_inventory.json) preserves counts by group and field without source prose.

**Next implementation choice:** add STL/BLK/TO to the sampled world and build validators for relations whose truth can be computed from the structured record—team/game scoring rank, bench/starter status, win/loss role, and double-double—before trying to reuse human clauses. Keep unsupported history and qualitative claims out unless separately modeled. Evaluate *semantically accepted* sentence yield and task-window form after each world/validator expansion; do not train models or claim realism based on the numeric opportunity count alone.
