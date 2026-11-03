# A shared-game genre mixture broadens tasks, but not realistic form

The earlier [run-syntax audit](SPORTS_RUN_SYNTAX_AND_VISIBILITY.md) found that retrospective articles rarely print the score snapshots needed to infer an in-game `X–Y` run from the visible prefix. A live score ledger can expose that history. This pilot therefore samples **two observation regimes over the same real game**: a retrospective report with final score, player and period facts, and an optional run mention; and a live ledger with printed score updates, a latest-score copy site, and two run-arithmetic sites. The goal is to separate *latent facts* from *visible predictive tasks* while broadening the DGP's task and genre coverage.

[`sports_genre_mixture_v1.py`](sports_genre_mixture_v1.py) draws a game eligible for both renderers within a stable train/heldout game partition, creates both views, then samples one view for the reference mixture (25% live). The paired alternative is available for controlled evaluation. The two views share the game ID, teams, outcome, and underlying play-by-play archive, but may describe **different scoring intervals**. No claim of a single shared run episode is made. The live sampler now accepts a game ID so pairing cannot silently substitute another game.

The [audit](audit_sports_genre_mixture_v1.py) sampled 100 game views per RotoWire/SBNation × train/heldout arm. It verified all 400 retrospective worlds, including result and player facts, the marked run sites from their visible score snapshots, and all 1,200 live score-copy/run sites from visible ledger lines. Train and heldout eligible game IDs are disjoint: 809/215 for RotoWire and 540/145 for SBNation. The selected genre counts are 26/25/29/31 live in the four 100-document panels. At the reference rate, retrospective run sites appear in only 6/1/15/16 selected documents; the live copy and run sites appear in every selected live document. This makes the observation-regime contrast explicit, but it gives sparse RotoWire retrospective-run evaluation at this panel size.

The retrospective **first 30 words** remain far from human reports sampled from the corresponding source and game partition. Human panels are matched to each generated retrospective panel's document count.

| Source / split | Generated retrospective docs | Generated top-ten word share | Human top-ten share | Generated cross-doc repeated-trigram share | Human repeated-trigram share |
|---|---:|---:|---:|---:|---:|
| RotoWire train | 74 | .357 | .286 | .609 | .238 |
| RotoWire heldout | 75 | .362 | .294 | .644 | .217 |
| SBNation train | 71 | .357 | .243 | .601 | .047 |
| SBNation heldout | 69 | .369 | .239 | .670 | .037 |

The live panel is a separate genre with no matched natural live-feed reference; comparing its aggregate form against retrospective articles would be misleading. Its `Live score`/`Score`/`Play` wrapper is visibly formulaic, and the earlier live-ledger audit found this too. The retrospective renderer likewise still narrates a small inventory of result, player, and period sentences. Simply mixing the two produces a more varied *aggregate* but does not fix either conditional distribution. It would also let a trainer exploit genre markers, so a mixture-level Zipf or NLL improvement alone would not demonstrate realistic language or reusable computation.

**Decision:** retain this as a shared-world, genre-conditioned task fixture. Do not train it as the accepted improved-language DGP or interpret its weights as a realism result. For the live arm, randomize or isolate run questions so the second answer cannot copy the first; the [longer live-ledger test](SPORTS_LIVE_LEDGER_3000_AND_TEACHER_FORCING.md) found that shortcut. For retrospective prose, add independently verified discourse acts and richer fact composition rather than more fixed player-stat or record paraphrases. Re-audit complete documents **and marked task windows** per genre, against an appropriate heldout human source where one exists, before training multiple bilinear models and comparing sparse/canonical methods. The broader sampled grammar with known local and long-range rules remains a parallel target; empirical carrier text here has no full exact next-token oracle.

The [aggregate JSON](results/sports_genre_mixture_v1_n100.json) contains game counts and form metrics, without source prose or play descriptions.
