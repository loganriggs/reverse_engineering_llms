# Two learned rules compose in heldout natural reports

This pilot combines two different visible computations in human RotoWire reports: a score sentence reveals the winner's final score, which is copied after intervening prose, and a parenthetical opener determines a later closer. The 2×2 controlled edits increment the winning score by one and switch round/square brackets independently. The human prose is a carrier; it has no complete symbolic next-token oracle. The score lead and later answer sentence are inserted with corpus-derived/limited natural-style forms.

The audit checks the visible score answer and unmatched bracket, nonoverlapping site spans, and independent answer changes across all four arms. Among 100 sampled unique train and 30 heldout games, 78 and 26 complete factorials have both targets as single BPE tokens inside a 256-token context. The same-game first-100-word repeated-trigram share is .182 original human report versus .212 edited score-copy carrier in train and .089 versus .125 heldout. The 41-word score-copy answer window repeats at .248/.190. Thus the carrier is much more varied than the fully synthetic event templates, but inserting the score lead/query measurably increases repetition. This is a useful bridge, not a passed natural-language realism gate.

Training only on the small score+bracket intersection performs poorly. With 800 repeated four-arm plans from 153–154 distinct games and 3,000 updates, two 2-block bilinear models reach paired bracket .679/.804 and paired score-copy .250/.179 on the same fresh 28-game heldout panel. A margin-subtraction variant has 0 paired arithmetic accuracy in one 16-game pilot, consistent with the earlier score-only carrier result; no margin circuit claim follows.

The stronger training distribution samples score-copy reports from the wider score carrier pool and bracket reports from the wider bracket pool. Each rule has 1,500 paired plans; **no jointly edited report enters training**. The two runs cover 526/521 distinct score games and 621/614 distinct bracket games. Models train with all-token next-token cross-entropy plus site weights for 6,000 updates. The joint test uses 28 identical, game-disjoint heldout reports for both models, all four edit arms.

| Training seed | Bracket paired exact | Score-copy paired exact | Bracket edit direction | Score edit direction | Other-rule TV (bracket / score) |
|---|---:|---:|---:|---:|---:|
| 55801 | .964 | 1.000 | 1.000 | 1.000 | .006 / .006 |
| 55802 | .911 | .929 | 1.000 | 1.000 | .015 / .007 |

This establishes behavioral competence on both rules in the combined format and low average interference under the two edits. The test is a heldout **combination of supervised rule sites**, not a proof that the model has two separable internal circuits. Score-only training documents can contain unedited brackets, and bracket-only reports can mention scores; what is withheld is the joint score-query-plus-bracket-edit factorial. The formats also differ, so failure would have mixed composition and style-transfer causes. The successful behavior makes these checkpoints suitable for testing whether weight-based reductions preserve both rules on unseen combinations.

Scripts: `sports_score_bracket_dgp.py`, `audit_sports_score_bracket.py`, `train_sports_score_bracket.py`, `train_sports_score_bracket_mixture.py`, and `eval_sports_score_bracket_transfer.py`. Aggregate audits, pilots, and fresh evaluations are in `results/sports_score_bracket_*.json`; model checkpoints and raw source prose remain local.
