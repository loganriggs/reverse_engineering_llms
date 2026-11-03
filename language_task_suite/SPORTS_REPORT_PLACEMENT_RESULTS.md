# Placement variation in the multirule sports-report pilot

The [multirule pilot](SPORTS_REPORT_MULTIRULE_PILOT.md) gave the model a real game-report carrier plus controlled score and player-stat facts. This follow-up varies **where** the player fact and three questions occur. For each paired document, the sampled game, human carrier sentences, controlled sentences, question forms, and answers are identical; only sentence order changes. The score remains first and the player fact precedes its question. This is a data-generating-process hyperparameter test, not a claim that a sentence shuffle produces natural discourse.

The [interleaved renderer](sports_report_multirule_interleaved.py) samples positions deterministically from a plan identity that excludes the edited numeric values. Consequently, score and player edits keep all sentence positions fixed. A stable hash also selects fixed or spread placement in the mixed arm. The [audit](audit_multirule_placement.py) checked 300 sampled plans in each game-key train/held-out group. Each group passed 8,100 visible-prefix answer checks, 900 selective-edit triplets across fixed, spread, and mixed, and 300 paired sentence-multiset comparisons; all 300 paired orders differed. Mixed assignments were 163/300 spread in train and 165/300 in held-out. The human carrier still lacks a full symbolic next-token oracle. [Aggregate audit](results/sports_report_multirule_placement_n300.json).

## Form result

Repeated-across-document trigram occurrence share on 64 unique games per group:

| Panel | Natural report | Fixed placement | Spread placement |
| --- | ---: | ---: | ---: |
| Train, first 100 words | .190 | .200 | .271 |
| Held-out, first 100 words | .178 | .198 | .273 |
| Train, full generated versus length-matched natural prefix | .210 | .299 | .286 |
| Held-out, full generated versus length-matched natural prefix | .202 | .297 | .287 |

The spread arm improves full-document repetition slightly, but makes the first 100 words worse because questions can appear early. The more decisive change is at the answer sites:

| 41-word answer window | Fixed, train | Spread, train | Fixed, held-out | Spread, held-out | Position-matched natural, held-out |
| --- | ---: | ---: | ---: | ---: | ---: |
| Winning score | .642 | .247 | .629 | .255 | .113 |
| Winning margin | .617 | .268 | .628 | .272 | .085 |
| Player points | .643 | .234 | .637 | .209 | .119 |

These natural windows are matched by relative position, **not** by task content. The spread arm remains more repetitive than the natural comparison, and a single genre cannot establish similarity to broader web language. The result isolates a substantial fixed-placement artifact, not a realism pass.

## Behavioral transfer

Two fixed-placement and two spread-placement width-64, four-head, two-block bilinear models were trained for 3,000 full-next-token CE steps with weight 500 at the three answer sites. The [cross-placement evaluator](eval_multirule_placement_transfer.py) uses the same 100 unique held-out game plans and sentence multisets in both layouts. The measure below requires the original **and** edited answer to be exactly top-1 correct. The margin/subtraction rule was not learned in either layout.

| Training layout, seed | Fixed score | Fixed player | Spread score | Spread player |
| --- | ---: | ---: | ---: | ---: |
| Fixed 96701 | .84 | .93 | .39 | .07 |
| Fixed 96702 | .87 | .79 | .48 | .05 |
| Spread 97301 | .77 | .18 | .85 | .68 |
| Spread 97302 | .79 | .07 | .78 | .49 |

Layout transfer fails especially for player copying even though the underlying world and sentence inventory do not change. Score copying transfers better from spread to fixed, perhaps because the score source remains at the beginning in both arms; this is an inference, not a circuit result. The paired edit gate supports looking for score and player-copy structure within the appropriate trained layout; it does not support looking for a learned margin circuit. [Aggregate transfer results](results/sports_report_multirule_placement_transfer_n100.json).

A mixed-layout model uses a stable hash to choose fixed or spread placement for each training plan, and trains for 6,000 steps so that approximate exposure **per layout** matches the 3,000-step single-layout arms.

| Mixed seed | Fixed score | Fixed player | Spread score | Spread player |
| --- | ---: | ---: | ---: | ---: |
| 97401 | .89 | .64 | .90 | .57 |
| 97402 | .90 | .27 | .90 | .25 |

Both seeds copy scores accurately in both layouts. Player-copy performance is much more balanced across layouts than in the single-layout arms, but the second seed is weak on that rule in **both** layouts. Margin remains at 0–.04. Thus mixed exposure reduces the severe layout dependence, but does not reliably produce a competent player-copy model. Total optimization steps differ, so this is a practical training comparison rather than an isolated estimate of the effect of mixed data.

This study is a bridge toward the main objective: vary the **data-generating process** while preserving known rules, train multiple model settings, and compare weight-only, sparse, and causal interpretability methods only on rules the model behaviorally learned. The next generator should move beyond shuffled sentences toward a sampled discourse plan with fact order, reference choices, distractors, and multiple safe realizations. The stopping criterion for this placement arm is a validated semantic/form audit plus paired-layout model transfer; more optimization of this single carrier is unlikely to close the remaining task-window and discourse gap.

For interpretability evaluation, the first mixed model supplies a more demanding, albeit imperfect, positive control than the fixed-only model: a proposed weight component for score or player copying should predict the correct source and survive answer-preserving sentence moves in both layouts. Causal ablation or path replay should affect the corresponding edit response in both layouts while sparing the other rule. The second mixed seed qualifies for a score-copy analysis, but its weak player-copy behavior should preclude a claimed player circuit. A method that reconstructs weights or logits but cannot recover cross-layout learned rule structure has not passed the intended discovery test. Winning margin remains a negative control until a model learns it behaviorally.
