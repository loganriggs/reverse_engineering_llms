# Two controlled rules in one natural sports report

The current research target is a variable sampled language distribution whose rules can be checked independently, then used to test whether weight-based methods recover the structure the model actually learned. Capability is a behavioral gate for interpretation, not the objective by itself. This pilot uses full human-written RotoWire reports as a natural-form carrier and makes two already occurring dependencies editable in the *same* report: a player's full name followed by a later surname, and an opening parenthesis followed by its matching closer.

## Why this pairing

The earlier score/player/alias proposal has poor *safe joint* yield. In 2,690 game-grouped training reports, 683 contain broad surface matches to all three, but none pass both the strict first-score and whole-clause player guards; the same is true in 708 held-out reports. Broad matches often add unsupported or contradictory prose. The [yield audit](audit_joint_natural_task_yield.py) records the distinction; it does not certify any full report as generated from a known world.

For the alias+bracket pairing, 741 of 937 alias-eligible training reports and 195 of 256 held-out reports contain a short parenthetical in the first 250 surface tokens. The generator samples one report, selects both sites, and creates a 2×2 factorial: alias A/B × round/square bracket. In the original arm, both sites are natural occurrences. The square arm is a controlled punctuation intervention. Every arm retains the surrounding human prose and changes only the selected factors. Its full next-token law is unknown: the two edited sites have visible-prefix rules, while the rest of the human text is a carrier.

## Checks before training

The [100-game audit](audit_sports_alias_bracket.py) separately samples train and held-out game groups. Both visible answers and the selective 2×2 edits pass 800 site checks per group, and all 100 factorials per group have the expected independent target changes. With GPT-2 BPE and a 320-token context, both sites are single-token and in context for 352/400 training arms and 336/400 held-out arms. The trainer filters to complete aligned factorials rather than silently dropping a difficult arm.

On 64 unique-game first-100-word panels, cross-document repeated-trigram share is 19.2% in the original training reports, 19.0% after alias edits and 19.0% after alias plus square-bracket edits; corresponding held-out values are 17.5%, 17.4%, and 17.4%. These are nearly the original report form, though the sports genre is not a matched FineWeb sample. A 41-word window around the two sites likewise changes little. See [aggregate audit JSON](results/sports_alias_bracket_audit_n100.json); raw report text is kept local.

## Behavioral and method gate

The [trainer](train_sports_alias_bracket.py) fits a bilinear transformer on whole report prefixes, with extra loss weight at the two rule sites. Held-out *game* groups have no overlap with training. The decisive metrics are exact full-vocabulary and two-candidate accuracy on both arms of each edit, edit direction, and whether each rule's output changes more under its own edit than the other rule's edit. Aggregate token loss alone cannot show that either rule was learned. Weight-only, sparse, and causal comparisons are warranted for a rule only after it passes these held-out behavioral checks; an unlearned rule is a negative control.

Two 64-wide, four-head, two-block bilinear transformers with RMSNorm were trained for 6,000 AdamW updates on 2,000 sampled training factorials per seed. Each factorial contributes four full-report arms. The aliases use the uniform eligible roster surname pool; alias and bracket sites receive loss weights 500 and 100 respectively. Seed 98201 also has a 3,000-step reading from its resumed trajectory. The [common fresh evaluator](eval_sports_alias_bracket_fresh.py) draws 100 new factorials on unique held-out games and scores both checkpoints on the same plans. These games do not overlap training, although they may overlap earlier held-out validation/test games.

| Fresh 100-game panel | Alias paired full-vocab top-1 | Alias paired two-candidate rank | Alias edit direction | Bracket paired full-vocab top-1 | Bracket paired two-candidate rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seed 98201 | .230 | .680 | .970 | .945 | 1.000 |
| Seed 98202 | .235 | .700 | .945 | .970 | .990 |

Bracket closure is a learned positive control in both seeds. Its own-edit output TV is .968/.979, versus .015/.004 under the alias edit, so it is largely specific to the bracket factor. Alias target-edit TV is .521/.542, versus .042/.048 under the bracket edit; the edit-direction and two-choice scores show genuine partial alias use. But exact paired open-vocabulary copying is only about 23%, and the answer-site NLL remains 5.8/6.8. Do not treat alias as a fully recovered computation or use these models to validate a claimed exact name-copy circuit. The [aggregate panel](results/sports_alias_bracket_fresh_seed98301_n100.json) contains checkpoint hashes and game hashes but no raw report text. The [individual run JSONs](results/sports_alias_bracket_depth2_seed98201_6000.json) and [second seed](results/sports_alias_bracket_depth2_seed98202_6000.json) retain their separate validation/test checks. Model checkpoints are local, not committed to Git.

This is a narrow natural-form bridge, not yet a fully sampled multi-rule language DGP. A later generator should independently vary rule prevalence, alias ambiguity, bracket depth, task distance, lexicon, discourse order, and genre, with semantics and visible-prefix oracle audited before training. This bridge is useful for testing whether methods distinguish two simultaneous computations without the extreme repeated frames of the previous symbolic report grammars.
