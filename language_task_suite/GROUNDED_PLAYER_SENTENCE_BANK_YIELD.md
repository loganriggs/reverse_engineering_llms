# Natural player-stat sentence bank: strict current-world yield is too small

The V5 quarter-prose renderer still repeats task-bearing phrases. A corpus-derived renderer would be preferable if many human-written sentences could be safely re-instantiated under the sampled world. This [screen](audit_grounded_player_sentence_bank.py) measures how many player-stat sentences from the official RotoWire and SBNation reports are licensed by the **current** world fields: one player plus box-score points, rebounds, and assists.

The final screen requires exactly one full roster-player name, led by that player; a complete declarative sentence; every digit attached to a points/rebounds/assists unit and equal to that player's box score; no spelled-out numbers or other unlicensed words; and no context terms for season, streak, records, injury, shooting, overtime, or other facts absent from the plan. It also rejects another roster player's surname. These rules are deliberately conservative and **not a proof of full semantic fidelity**. They likely reject many valid sentences with complex syntax. The audit saves counts and hashes of slotted forms, not verbatim source prose.

| Corpus/game group | Reports | Candidate occurrences | Reports with a candidate | Reports with ≥2 candidates | Distinct slotted forms |
| --- | ---: | ---: | ---: | ---: | ---: |
| RotoWire train | 2,690 | 13 | 13 | 0 | 10 |
| RotoWire held-out | 708 | 2 | 2 | 0 | 2 |
| SBNation train | 6,065 | 33 | 33 | 0 | 22 |
| SBNation held-out | 1,568 | 11 | 11 | 0 | 6 |

Manual samples of accepted clauses are simple, such as a player tallying points and assists or finishing with points and rebounds. An earlier permissive screen appeared to offer thousands of candidates, but manual review found it admitted spelled-out extra statistics, steals and blocks, evaluative comparisons, injuries, and sentence fragments. It checked matching **digit** tokens while missing those other claims. The strict result supersedes that initial yield. It even found and rejected dangling fragments such as a sentence ending after “while.”

The result is a clear constraint on the next DGP: **we cannot build varied, grounded multi-fact reports by substituting names and numbers into this small safe bank.** No game has two accepted clauses. The low yield is partly due to conservative validation, but that itself identifies the engineering work: expand the sampled world to facts actually expressed in reports (shooting, steals, blocks, turnovers, quarter trajectory, and supported record/history fields) and validate their linguistic relations, or use another aligned source with richer semantic records. Merely collecting more player-stat sentence strings under a points/rebounds/assists-only world would reproduce the same unsupported-claim problem.

The [aggregate](results/grounded_player_sentence_bank_yield.json) provides group counts and hashed form frequencies. This is a source-yield decision, not a new training distribution or evidence that a model learned any rule. The broader research goal remains to train bilinear models on a variable sampled process and evaluate weight methods only after rule-specific behavioral gates.
