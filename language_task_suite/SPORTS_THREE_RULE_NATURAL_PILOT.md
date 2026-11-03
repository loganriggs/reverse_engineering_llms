# Three simultaneous rule sites in human sports reports

This is a corpus-anchored bridge toward a more varied, rule-bearing language distribution. Each plan selects a human RotoWire report with three nonoverlapping sites, then independently chooses two surname aliases, round/square brackets, and period-plus-capitalized-word/semicolon-plus-lowercase-word. All eight combinations share the same underlying report. This is **not** a fully sampled symbolic language and has no exact oracle for background next-token probabilities. The report text is not included in public artifacts.

## Data and checks

The three answer rules are: a later surname follows an earlier full name, a closer follows the unmatched visible bracket, and a word after a period starts uppercase whereas a semicolon continuation is lowercase. The last rule determines **case conditional on the lexical word**; punctuation alone does not determine which word should follow. The target words are strongly imbalanced: 71/100 train and 72/100 heldout audited plans use `The`. This makes full-vocabulary accuracy partly a lexical-frequency test.

An audit of 100 unique games per source split checks 2,400 answer spans and visible-prefix rules, plus independent changes at each site across the 2×2×2 cube. All eight arms are single-BPE and within 320-token context for 90/100 train and 88/100 heldout plans. Fully edited reports closely preserve first-100-word form: train cross-document repeated-trigram share .174 original/.174 edited; heldout .184/.183. Capitalization-site 41-word windows are .113/.112 and .109/.108. These genre-specific natural reports remain narrower than broad FineWeb. Semicolon edits need more manual semantic review; the existing check certifies surface orthography, not every clause's meaning.

## Bilinear model behavior

Both runs use a two-block, width-64, four-head bilinear transformer with RMSNorm, 320-token context, ordinary next-token cross-entropy on **every nonpadding token**, and extra weights at the three sites (alias 500, bracket 100, capitalization 100). Game-disjoint heldout evaluation uses 40 plans × eight arms per seed. All eight combinations occur in training, so these results do not establish whole-combination generalization.

| Run | Train plans / unique games | Steps | Bracket paired exact | Alias paired exact | Capitalization paired exact |
|---|---:|---:|---:|---:|---:|
| 55301 | 1,000 / 523 | 3,000 | .950 | .000 | .419 |
| 55302 | 2,000 / 625 | 6,000 | 1.000 | .100 | .506 |

The bracket rule is a strong behavioral positive control. Alias candidate-pair success is only .081/.488, with full-vocabulary paired exact .000/.100; it is not ready for weight-based circuit claims. Increasing data and updates did not make it robust. The case candidate-pair success is .975/1.000 and the punctuation edit changes the distribution strongly, but the model often predicts `The`/`the` regardless of the true word. In seed 55302, the semicolon arm gets `the` right 107/120 times and gets `a`, `he`, and `it` right 0/4, 0/8, and 0/28. The period arm gets `The` right 94/120, with zero correct on those other words. Thus case response is visible while lexical choice is not solved.

The per-seed comparisons are descriptive, not a controlled ablation against the earlier two-rule model: game panels and seeds differ. The earlier two-rule 6,000-step runs attained alias paired exact .125/.213 and bracket .875/.975 on their own heldout panels. A matched-panel ablation would be needed to attribute any alias decline to the third rule.

## Decision

Use bracket closure as the positive control for the next weight-based canonicalization/sparsification comparison, and alias as a negative control. Treat capitalization as a conditional case probe, not a third competent full-vocabulary task. Before claiming a three-task shared circuit, redesign the third rule so its target is identifiable from the visible prefix without a dominant-word shortcut, and evaluate on balanced lexical targets and whole-combination holdouts. Stop simply scaling this same imbalanced three-rule sampler; another increase in updates alone would not test the missing inference.

Reproduce with `audit_sports_alias_bracket_caps.py`, `train_sports_alias_bracket_caps.py`, and `diagnose_sports_alias_bracket_caps.py`. Aggregate JSON results are in `results/sports_alias_bracket_caps_*`; model checkpoints remain local.
