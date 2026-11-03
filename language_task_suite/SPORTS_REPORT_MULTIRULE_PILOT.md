# Two learned rules in one natural-carrier bilinear model; form gap remains

This pilot moves beyond one controlled score question per report. The [generator](sports_report_multirule_dgp.py) samples a real RotoWire game report as an empirical carrier, inserts a score statement and a box-score-grounded player tally, and asks three questions in random order: winning-score copy, winning margin, and the named player's point total. Its 78 usable tally templates are extracted only from the [train-group complete-clause bank](PLAYER_STAT_GRAMMAR_BANK_AUDIT.md), then re-instantiated for a player whose surname is absent from the retained carrier. A score +1 edit changes score copy and margin; a player-points +1 edit changes only player copy. The player is chosen below the team maximum so the edit cannot overturn another player's team-high status. Only these inserted facts and answer sites have known rules; the human carrier has **no full next-token oracle**.

The two successfully learned rules are both **copying**. The score source precedes roughly 100 carrier words; the player tally is inserted close to the questions. This pilot tests whether the model separates two sources in a shared document, not whether it learned rich language semantics or a difficult long-range player relation.

## Semantic and form gates

The [audit](audit_sports_report_multirule.py) sampled 300 documents in each game-key train/held-out group and checked original, score-edited, and player-edited versions: **2,700 visible-answer checks and 300 selective-edit triplets per group**, all passed. The answer parser reads the score from the first rendered sentence and the player name from the visible question, then locates the completed player source statement in the prefix. All six question orders occur. The eligible carrier pool has 554 train and 123 held-out game reports; the player selector found compatible unmentioned players in sampled cases. The audit saves aggregate statistics and game-key hashes, not source prose.

The **first 100 words** of the generated documents resemble the paired originals on repeated-cross-document trigram share: RotoWire train .199 versus .186, held-out .200 versus .178. That favorable crop does not include all questions. On **full generated documents** versus length-matched prefixes of their paired reports, the shares are .296 versus .211 (train) and .294 versus .199 (held-out), with generated top-ten word shares .288/.286 versus .258/.257. The 41-word answer windows repeat much more: .608–.664 across the three rules on train and .625–.647 on held-out. Thus natural carrier prose helps whole-document form, but the **task-bearing language still fails the realism gate**. This is a useful multi-rule bridge, not the desired realistic sampled language DGP. [Aggregate form and semantic results](results/sports_report_multirule_audit_n300.json).

## Bilinear-model behavioral gate

Two width-64, four-head, two-block bilinear transformers with RMSNorm were trained for 3,000 full-next-token CE steps with weight 500 on each of the three answer sites. Each seed drew 2,000 training triplets from 537 distinct train-group games; validation and test games are in the disjoint held-out group. Training uses GPT-2 byte-pair tokens. On a new panel of 100 unique held-out games per seed, the decisive measure is **both original and edited answers exactly top-1 correct**. The fresh panel uses new form/question draws; its games can overlap earlier held-out evaluations, but none is a training-group game.

| Seed | Winning-score paired top-1 | Player-points paired top-1 | Margin paired top-1 | Score/player targeted TV | Score/player nuisance TV |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96701 | .90 | .92 | .00 | .978 / .948 | .005 / .019 |
| 96702 | .89 | .79 | .02 | .982 / .900 | .008 / .089 |

TV is full-vocabulary output-distribution total variation at the answer position. Targeted edits also move the correct-answer logit preference in the expected direction in 98–100% of cases for the two copy rules. Margin direction is near chance and exact paired performance is 0–2%; **no margin circuit is eligible for interpretation**. The copy rules are behaviorally learned in both seeds, though player-copy exactness is weaker in seed 96702. Aggregate [seed 96701 training](results/sports_report_multirule_depth2_3000.json), [fresh behavior](results/sports_report_multirule_depth2_3000_fresh100.json), [seed 96702 training](results/sports_report_multirule_depth2_seed96702_3000.json), and [fresh behavior](results/sports_report_multirule_depth2_seed96702_fresh100.json) are committed. Model tensors and optimizer caches stay local. Seed 96701 was resumed at step 1,500; its 1,500-step model file was overwritten on resumption, so the final 3,000-step checkpoint, not that intermediate, supports the interpretation result.

## First task-blind weight probe

At each attention head, compute the product of the Frobenius norms of the two QK matrices and the OV matrix. This uses **weights only**, no training or test examples. Rank all eight heads before examining task labels. On a separate 100-game held-out draw, ablate each whole head and measure loss of paired exactness. Both seeds put the score-copy-critical head **first** and the player-copy-critical head **second** by this simple weight score:

| Seed | Weight rank 1: score-copy head | Score paired-accuracy drop | Weight rank 2: player-copy head | Player paired-accuracy drop | Cross-rule drop from those heads |
| --- | --- | ---: | --- | ---: | ---: |
| 96701 | layer 2, head 3 | .90 | layer 2, head 1 | .85 | .00 / .00 |
| 96702 | layer 2, head 2 | .93 | layer 2, head 1 | .71 | .00 / .01 |

The head index for score copying changes across seeds. Other heads can also matter, especially first-layer heads for player copying. Ablation is a causal necessity test in the intact network, not proof of a unique complete circuit. The weight product is gauge-sensitive, ignores RoPE and actual contexts, and does not itself tell us *which* rule a large head serves. The result is nevertheless a replicated **task-blind nomination** of two causally important heads in one multi-rule model. See both [head-probe aggregates](results/sports_report_multirule_depth2_heads_n100.json) and [seed-2 head probe](results/sports_report_multirule_depth2_seed96702_heads_n100.json).

## Decision

Use the two copy rules as a bounded positive control for comparing function-level tensor reductions, data-free versus distribution-weighted priors, and sparse-path objectives at matched predictive and causal fidelity. Keep score and player interventions separate, and hold one rule's cases out of any data-aware fit when testing transfer. Do **not** interpret margin or treat the current head score as a canonical attention decomposition. In parallel, the primary DGP work must replace the repeated answer-question grammar with document-level semantic realization and audit both full text and task windows before claiming a more realistic language distribution.
