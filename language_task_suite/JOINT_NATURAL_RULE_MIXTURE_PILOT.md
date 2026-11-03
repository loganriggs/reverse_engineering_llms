# Joint natural-text rule mixture: third rule remains unlearned

This is a bridge toward diverse, realistic task coverage, **not** the final fully sampled language DGP. It combines a human sports-report carrier with independently edited full-name→later-surname reuse and parenthetical closure, and a separate FineWeb carrier with a sampled document-wide US/UK spelling convention. Only the edited sites have controlled rules; background prose has no symbolic next-token oracle. The sports rules co-occur in one report. The spelling rule is mixed across documents, not inserted into the same sports reports.

The [joint continuation script](finetune_joint_natural_rules.py) starts from the existing 6,000-step depth-2, width-64, four-head bilinear sports checkpoint (seed 98201). It compares 1,000 updates of equal-size minibatches: four sports and four spelling windows versus eight sports windows. Each source example receives mean token cross-entropy with its site weights (alias 500, bracket 100, spelling 100). The two arms have the same update count and starting weights, but the sports-only arm sees twice as many sports examples. The train set uses 1,000 sampled sports worlds and 1,970 FineWeb spelling windows. Evaluation uses 40 heldout-game sports worlds and 215 heldout FineWeb document pairs. No sports game overlaps training and evaluation. The initial and final metrics use the same heldout panels.

| Checkpoint | Sports token NLL | FineWeb token NLL | Bracket paired exact | Alias paired exact | Alias paired two-candidate | Spelling paired two-candidate |
|---|---:|---:|---:|---:|---:|---:|
| Starting sports model | 4.700 | 13.067 | .950 | .200 | .700 | .014 |
| Mixed continuation | 4.947 | 8.038 | .950 | .275 | .725 | .014 |
| Sports-only continuation | 4.712 | 12.985 | .950 | .212 | .725 | .014 |

Bracket's paired candidate ranking and edit direction remain 1.0 in all three checkpoints on this panel. The mixed model's spelling cue-edit direction is .498, versus .474 at the start and .465 in the sports-only control—near chance. Its US/UK full-vocabulary top-1 values rise only to .056/.028. The apparent alias gain is 40-game, one-seed exploratory evidence and does not establish positive transfer. Mixed training raises sports token NLL while lowering FineWeb token NLL. These are distinct from rule-site and causal outcomes.

**Decision:** this mixture creates a common model exposed to three controlled rule families and two natural text sources, but only bracket closure is strongly learned. The alias remains partial and spelling remains a negative control. It would be invalid to search the mixed model's weights for a claimed spelling circuit. The result also agrees with the earlier [standalone spelling pilot](CORPUS_SPELLING_STYLE_PILOT.md), where stronger site weighting and paired augmentation did not yield heldout cue use. More training on this tiny spelling bank is a poor next bet; the next dataset improvement should increase distinct grounded contexts and rule-bearing windows while tracking their natural rates and form. Keep whole-document/source holdouts and test paired edits before any canonicalization or sparse-weight method is credited with recovering a rule.

The [aggregate](results/joint_natural_rules_seed41501_1000.json) is public; the teacher and new model checkpoints, FineWeb token arrays, and source reports stay local.
