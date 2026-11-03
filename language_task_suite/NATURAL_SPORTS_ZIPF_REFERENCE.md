# Natural sports reports as a second frequency reference

The [reproducible audit](audit_natural_sports_zipf.py) samples **480 game
keys shared by RotoWire and SBNation training reports**, requiring at least
100 alphabetic words in both reports. It compares the first 100 lowercase
words under the same regex as the V7/FineWeb rank audit. There are 1,755
eligible paired games. The [aggregate result](results/natural_sports_zipf_n480.json)
contains rank counts, source hashes, and game-key hashes, but no report text.

| 48,000-word panel | Distinct words | `the` share | Top-ten share | Singleton types | Rank 10–100 slope |
|---|---:|---:|---:|---:|---:|
| FineWeb, frozen broad-web panel | 8,975 | 4.86% | 21.41% | 4,912 | −.948 |
| Human RotoWire, paired games | 3,159 | 9.40% | 26.27% | 1,299 | −.816 |
| Human SBNation, paired games | 4,630 | 8.58% | 25.10% | 2,232 | −.854 |
| V7 long-document sample, separate worlds | 438 | 16.41% | 29.94% | 2 | −.551 |

The two human sports styles have more common-word concentration and a
smaller vocabulary than broad FineWeb. SBNation is lexically broader than
RotoWire **on these paired games**. V7 is substantially narrower than both,
especially in its near-absent singleton tail. The human pair controls game
identity, but not author, sentence content, or source editorial style. V7
and FineWeb use independent panels, so their rows are scale comparisons, not
paired causal effects.

This changes how to judge a sports-centered generator. Matching FineWeb's
global `the` rate would be the wrong single target for a sports-report arm;
the within-genre human panel is a more appropriate lexical reference. A
larger multi-genre synthetic distribution can separately target a broader
web mixture. None of these rank curves establishes coherent semantics,
natural task prevalence, or an exact next-token law. Natural carrier text
also lacks a full known oracle; only its controlled rule sites are labeled.

For the next generator, use these panels as two distinct targets: a typed
sports/report regime with verified factual and rule semantics, and a broader
web-style regime with more document forms and a larger lexical tail. Keep
whole-game and whole-task holdouts. Train small bilinear models only after
each regime passes its own form and answer audits, then interpret only rules
that pass behavioral and causal edit gates.
