## H975 — Prospective composition panel yields seven new native-competent cohorts

Previous turn was progress: H974 rescue tests established a partial composition/transport route. Candidate sites and formulas were frozen before this panel: adjective-last-subtoken MLP3 local/total interaction, period MLP4 write rescue, attention5 transport. Sixteen adjective pairs (four anchors, twelve new to H962/H966 panels), four templates (film/movie/book/Overall sentiment wording), four factorial states produce64 quartets/256 unique prompts. Four predeclared lexicon cohorts of four pairs; no outcome-based replacement.

Gate unchanged: >=4 quartets, >=75% all-four binary correct, >=50% pooled full-vocabulary intended-group top1. Eight of16 cohorts pass: original film anchors; film new3; movie anchors and new3; book anchors and allthree new lexicon cohorts. Seven passing cohorts therefore supply contexts beyond the original anchor setting, with vocabulary shared across template variants rather than seven independent lexical datasets. Sixteen exact anchor prompts overlap earlier panels;240 are new relative to those checked panels.

New1 adjective pairs: amazing/horrendous, outstanding/dismal, marvelous/mediocre, exceptional/lousy. New2: splendid/ghastly, magnificent/appalling, terrific/deplorable, fabulous/pathetic. New3: pleasing/distasteful, admirable/abominable, entertaining/dull, compelling/unwatchable. Novelty is relative to the checked research panels, not all historical work or model training.

Allfour book cohorts pass. Book/new1 is especially useful:100% all-four binary correctness,68.75% full-vocabulary top1, allfour adjective pairs single-token. New2 and new3 include multi-subtoken adjectives; the frozen last-subtoken alignment is retained and does not claim to represent every adjective token.

All Overall-sentiment cohorts achieve100% all-four binary correctness but fail full-vocabulary top1 (0–12.5%). They remain diagnostic rather than successful native task-transfer examples. Film/movie failures mostly involve negated-negative endpoints and/or insufficient full-vocabulary top1. Semantic reversal of not-negative remains an operational expectation, not a universal linguistic truth.

Audits:257/257 forward examples; repeat logit replay1.43e-5; manifest/input hashes unchanged;256 distinct prompts; no tokenization exclusions;16 exact prior prompt overlaps; finite outputs. Native logits and locked-site activations saved for follow-up. No interventions, training, or reserved-test access. Natural-document pool26,112.

Next: apply frozen local/total MLP3 intervention formulas and periodMLP4 rescue to every cohort without site retuning. New teacher quartet coefficients are permitted by the assay but are not frozen learned features; distinguish mechanistic-site transfer from encoder generalization. H961 context-bridge predictor evaluation remains pending.
