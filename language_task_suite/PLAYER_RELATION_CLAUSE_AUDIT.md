# Adding box-score relations barely grows the narrow natural clause bank

The [complete player-stat tally screen](PLAYER_STAT_GRAMMAR_BANK_AUDIT.md) admitted 153 RotoWire and 67 SBNation training clauses. The source records contain team city, final team scores, player minutes, starting position, and points. This [relation screen](audit_player_relation_clauses.py) asks how many more **complete, independently checkable clauses** appear when those fields license a win/loss suffix, an off-the-bench suffix, or a team/game-high scoring claim. It preserves the same game-key train/held-out split and requires the base tally to pass all stat and number checks.

| Source/group | Additional clauses | Games with one | Bench | In win | Team-high | Game-high |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RotoWire train | 9 | 9 | 5 | 1 | 2 | 1 |
| RotoWire held-out | 1 | 1 | 1 | 0 | 0 | 0 |
| SBNation train | 8 | 8 | 7 | 1 | 0 | 0 |
| SBNation held-out | 0 | 0 | 0 | 0 | 0 | 0 |

The outcome check links the named player to the record's team city and compares both final scores; tied or unlinked teams are rejected. Bench requires `START_POSITION=N/A` **and positive numeric minutes**, distinguishing a participant from an inactive player. Rank checks the player's points against all recorded scorers in the relevant team or game; missing points for a player with numeric minutes make the check fail. A tie for the maximum is allowed as a high score. Adversarial fixtures covered correct/incorrect win and bench assertions and team/game-high rank. The [aggregate JSON](results/player_relation_clauses.json) stores counts and form hashes only; source examples were inspected locally and are not committed.

This is **not** an estimate that such semantic relations are rare in natural reports. The explicit grammar handles a few sentence shapes; many valid sentences express the same relations through varied syntax or mix them with additional facts. The result is narrower and more useful: adding a handful of licensed suffixes to a small tally grammar does **not** turn it into a variable document-level source. Across RotoWire train, the two screens together supply at most 162 clauses in 2,690 game reports, even before accounting for discourse coherence when independently resampled. Stop extending this bank with one-off phrase patterns. The next generator experiment should model or sample whole-document discourse acts and realization choices while validating every inserted claim from a richer structured world, then test task windows and full-document form on game-disjoint natural panels before training. Keep semantic acceptance distinct from numeric matching and from learned-model behavior.
