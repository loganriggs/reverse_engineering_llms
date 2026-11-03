# Typed human stat predicates: a partial form gain, harder model learning

The previous small-LM sentence proposals often introduced unsupported claims.
This pilot takes a narrower route: extract only a player-led statistic
**predicate** from real sports reports, discard every surrounding clause, and
reinstantiate that predicate using a sampled player's true box-score facts.
The bank contains no full human-written sentences. It is a candidate surface
renderer, not an accepted realistic language distribution.

[`audit_player_predicate_fragments.py`](audit_player_predicate_fragments.py)
requires one unambiguous full roster-player name, a restricted factual verb,
a contiguous list of explicit stat quantities, and equality with that
player's box score. It rejects repeated fields and incomplete conjunctions.
It collects forms only from the stable **training-game** partition. The
aggregate output contains counts and hashes, never source prose.

| Source / partition | Fact-matched fragments | Games with a fragment | Distinct forms |
| --- | ---: | ---: | ---: |
| RotoWire train | 743 | 599 / 2,690 | 164 |
| RotoWire held out | 182 | 157 / 708 | 61 |
| SBNation train | 526 | 445 / 6,065 | 98 |
| SBNation held out | 147 | 127 / 1,568 | 40 |

The earlier whole-sentence guard accepted only 13 RotoWire training clauses.
The larger fragment yield is plausible because we remove contextual prose
such as historical comparisons, injury claims and outcome descriptions. It
does **not** license transplanting the original sentence. The current renderer
uses 95 distinct training-derived forms whose first stat is points and whose
fields are limited to points, rebounds and assists, matching the sampled
world's available facts and the independent visible-prefix scorer.

[`sports_empirical_stat_renderer.py`](sports_empirical_stat_renderer.py) is
an optional `empirical_stat_rate` arm of the sampled sports document process.
For a controlled form test, each of 128 worlds per source/game partition was
rendered once with the original fixed player forms and once with the typed
predicate bank. Both arms use five single-player clauses (`pair_rate=0`), rich
stats, and identical score/recap sampling controls. The audit independently
checked every emitted player-stat number against the sampled plan, all three
visible-prefix answers, and selective winner, margin, and top-player edits:
**128/128** checks of each edit and arm passed in each partition.

At 55 words per document, the form changed in the right direction but is
still far from human sports reports:

| Panel | Top-ten word mass: fixed → empirical → human | Repeated trigram share: fixed → empirical → human |
| --- | --- | --- |
| RotoWire train | .455 → .382 → .260 | .724 → .616 → .230 |
| RotoWire held out | .452 → .384 → .264 | .718 → .634 → .239 |
| SBNation train | .437 → .389 → .244 | .756 → .658 → .085 |
| SBNation held out | .449 → .394 → .242 | .757 → .670 → .097 |

The human comparison is descriptive: human reports are not paired by sampled
world and contain many discourse acts that this generator lacks. The fixed
and empirical arms are paired by world. The new bank raises word-type counts
by roughly 48–54 per 128-document panel but leaves repetitive score/recap
frames and document planning largely intact. It should be used as a
**controlled DGP surface hyperparameter**, not as a realism pass.

I trained matched depth-2, width-64, four-head bilinear models for 2,000
updates on 4,000 sampled documents each, ordinary next-token loss at every
non-padding position, seed 37301. No rule labels entered training. Held-out
games and 100 fresh documents gave:

| Training surface | Token NLL | Winner candidate accuracy / correct team-swap switches | Margin accuracy / score-edit switches | Top-player accuracy / points-edit switches |
| --- | ---: | ---: | ---: | ---: |
| Fixed forms | 1.047 | .95 / 81 | .09 / 1 | .24 / 1 |
| Empirical predicates | 1.277 | .63 / 22 | .06 / 0 | .14 / 1 |

The more variable arm is harder at this fixed budget; neither model learned
margin or top-player comparison. The empirical arm's winner behavior is too
weak for a weight-circuit claim. The fixed arm's winner behavior is eligible
only as a narrow positive control, not proof that it learned the intended
score-computation algorithm. One model seed is insufficient to rank surfaces
or training efficiency. Site accuracy ranks provided candidates, whereas
token NLL averages all next-token positions.

The next DGP iteration should vary the *document planner and recap language*
with equally explicit semantic contracts, not merely add more safe stat
verbs. A matched training sweep over surface rate, rule prevalence and model
budget can then test whether multiple rules remain learnable, after which
weight-based interpretation can be evaluated on each behaviorally learned
rule with whole-task and whole-style holdouts.

Reproduce the aggregate screens and training arms with
[`audit_sports_empirical_predicates.py`](audit_sports_empirical_predicates.py)
and [`train_sports_document_pilot.py`](train_sports_document_pilot.py).
The two saved 2,000-step checkpoints and JSON records are under
[`results`](results/).
