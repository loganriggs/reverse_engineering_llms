# A naturally worded entity-reference rule in sports reports

The previous [multirule placement study](SPORTS_REPORT_PLACEMENT_RESULTS.md) found that shuffling fixed task sentences lowered answer-window repetition but did not produce convincing task-bearing prose. This pilot instead starts from complete human-written RotoWire reports and uses a naturally occurring dependency: a player is introduced by full name, then referred to by surname later. It renames that player consistently in two counterfactual copies of the same report. The target is the first later surname at least 25 corpus tokens after the full-name source. This creates a controlled **entity-alias prediction** with natural surrounding syntax, without inserting a synthetic question sentence.

The [generator](sports_entity_alias_dgp.py) restricts the source to an exact two-word roster name appearing once, rejects an earlier standalone surname, same-surname roster collisions, a different full name with that surname anywhere in the report, and unedited inflected surname tokens. Both substitute surnames are drawn from train-group player surnames that tokenize as one GPT-2 byte-pair token after a space. The edit changes every exact occurrence of the chosen surname, including source and target, while all other text stays identical. This is a text-level alias intervention; the changed name need not match the real game's external roster. The original coreference relation is inferred by a conservative surface screen and spot inspection, **not** certified by a complete semantic parser.

The [audit](audit_sports_entity_alias.py) screens whole reports under the existing game-key split. Its repeated-number counts are intentionally only surface lower bounds: the exact score pair occurs in two separate sentences in 13/2,690 train reports and 0/708 held-out reports; a full player name with its point total occurs in two separate sentences in 8 and 5. By contrast, a full name followed at least 25 tokens later by the same standalone surname within the first 250 tokens survives the stricter alias filter in **937/2,690 train reports and 256/708 held-out reports** (1,219 and 306 candidate pairs). Thus natural name reference is much more available than natural exact-number repetition in this corpus. This supports adding entity reference as a task family; it does not estimate its rate in general web text.

On a 100-unique-game panel per split, all 200 paired substitutions preserved the non-alias text, exposed the edited full-name source and later surname target, and had a single-token target within 320 model tokens. A one-reader spot check of 24 fresh source/target snippets found a false same-surname match to another person; the extractor was tightened to reject other full-name uses. A separate edit renaming a *different* previously introduced player left the source and target unchanged on 61/100 train and 63/100 held-out panel plans. The remaining screen is still fallible, particularly for two people mentioned only by a shared surname. The audit saves aggregate numbers and game-key hashes, not source prose. [Aggregate audit](results/sports_entity_alias_audit_n100.json).

| 100-game form panel, repeated-across-document trigram occurrence share | Original prose | Alias-edited prose |
| --- | ---: | ---: |
| Train, first 100 words | .212 | .209 |
| Held-out, first 100 words | .200 | .197 |
| Train, 41-word target windows | .150 | .148 |
| Held-out, 41-word target windows | .173 | .171 |

The near-match is expected: almost all words are untouched human prose. It is stronger evidence about the **surface of this task site** than the earlier fixed query templates, but it says nothing about the full next-token oracle. Uniformly sampling among 209 one-token substitute surnames also makes the intervention's name distribution less natural than the source corpus.

The [name-prior calibration](calibrate_sports_alias_prior.py) makes this a tunable DGP parameter. In 2,000 train-group draws, 65.7% of eligible original target surnames are one-token aliases. Within that restricted reference distribution, the top-ten surnames take 40.1% of mass. The uniform roster pool gives only 8.5% and Jensen–Shannon divergence .334 from the restricted reference. Sampling train-group surnames that actually occur at eligible reference sites, with weight proportional to their observed frequency, gives 36.0% top-ten share and divergence .018. Exponent 1.25 gives a nearly identical divergence .018 but a more concentrated 45.2% top-ten share. The fitted-prior arm uses exponent 1.0 as the simpler default. Its separate 100-game per-split audit passes every paired substitution and one-token/context check; 77/100 train and 55/100 held-out plans also admit a nuisance rename that preserves the target. This matches only a **one-token surname marginal conditioned on this task**, not Zipf's law, document-level name burstiness, or the joint language distribution. [Aggregate prior sweep](results/sports_entity_alias_prior_n2000.json); [fitted-prior semantic/form audit](results/sports_entity_alias_eligible_audit_n100.json).

## Bilinear behavioral gate

Two width-64, four-head, two-block bilinear transformers with RMSNorm were trained on 2,000 paired train-group plans, ordinary next-token CE plus weight 500 on the later surname token. Model inputs include at most 320 GPT-2 BPE tokens. Each paired plan supplies two different edited surnames, so a model cannot rely on the original real-world name association. Validation and test use disjoint held-out game keys; the [fresh evaluator](eval_sports_entity_alias_fresh.py) uses the same 100 new game plans for all checkpoints. Its games may overlap earlier held-out evaluations but never training-group games. Two metrics answer different questions: **paired full-vocabulary top-1** requires both aliases to be the model's highest-probability token; **paired two-alias rank** only requires the correct alias to outrank the other controlled alias under each edit.

A direct causal-mask check replaced every input token after a scored site in a saved model; the answer-position logits were exactly unchanged in that check. This rules out a simple future-token leakage explanation for the reported site scores.

| Depth-2 seed, steps | Fresh paired full-vocab top-1 | Fresh paired two-alias rank | Correct edit direction |
| --- | ---: | ---: | ---: |
| 97601, 3,000 | .02 | .36 | .87 |
| 97601, 6,000 | .23 | .75 | .98 |
| 97602, 6,000 | .16 | .67 | .97 |

Both uniform-prior 6,000-step models are sensitive to the intervention and often prefer the appropriate alias in a two-choice comparison, but remain weak at exact open-vocabulary copying. The 3,000-step row is the same seed and data construction as the later 6,000-step row; it is a separately completed run, not a saved intermediate evaluation of the same optimizer trajectory. [Training and evaluation code](train_sports_entity_alias.py); [original fresh aggregate](results/sports_entity_alias_fresh_seed97701_n100.json).

On the **same 60 held-out games** that admit a nuisance edit, the mean probability drop for the original target surname under a target-entity rename versus a different-player rename is .245 versus .065 for seed 97601 and .134 versus .038 for seed 97602. Full output-distribution TV is .567 versus .413 and .485 versus .398. The target response is larger, but the nuisance response is substantial; the current models have not isolated a clean entity-specific computation. The nuisance subset is selected for having another name to edit and should not be presented as all-report prevalence.

## Name-prior transfer

One additional depth-2 model, seed 97603, was trained for 6,000 steps on the fitted eligible-target prior at exponent 1.0. The same three 6,000-step checkpoints were evaluated on a common 100-game **uniform** panel and separately on a common 100-game **fitted-prior** panel. Models share worlds and aliases *within* a panel; the two panels are separate draws, so their cells are not paired report by report.

| Training prior / seed | Uniform panel: paired top-1 / paired two-alias rank | Fitted-prior panel: paired top-1 / paired two-alias rank |
| --- | ---: | ---: |
| Uniform 97601 | .23 / .75 | .07 / .56 |
| Uniform 97602 | .16 / .67 | .08 / .47 |
| Fitted 97603 | .02 / .16 | .27 / .46 |

The fitted-prior model's higher in-prior exact score does **not** imply stronger general binding: its two-alias rank is below both uniform models on that panel, and it collapses on uniform aliases. The prior also narrows the effective answer vocabulary and changes label entropy; this experiment does not isolate frequency matching from those statistical differences. A context-free frequency baseline could not get *both* distinct aliases right in a pair, so the fitted model does respond to context, but its response is distribution-specific. [Uniform panel aggregate](results/sports_entity_alias_fresh_uniform_seed97701_n100.json); [fitted-prior panel aggregate](results/sports_entity_alias_fresh_eligible_seed97701_n100.json).

On the fitted-prior panel's same 64 nuisance-eligible games, seed 97603's mean probability drop for the original target is .240 under the target rename versus .029 under the different-player rename. This is more selective in that metric than the uniform models' corresponding .102/.043 and .093/.035 on this panel, although none of these local numbers establishes a unique circuit or complete distributional fidelity.

This is a task-rich natural-text bridge, not the final sampled toy language. The experiment has one controlled rule family, a surface-screened coreference target, and no exact next-token law for the human report. Its full-vocabulary gate and prior transfer are too weak for confident weight-circuit recovery, and its nontrivial nuisance effects add caution. The broader DGP should combine this kind of natural entity reference with other sampled world-state rules, while varying rule prevalence, name frequency, reference distance, distractors, and realization style. A weight-method comparison should use both name priors and report rule prediction, intervention specificity, and genuine transfer separately from reconstruction. Only rules that pass behavioral and causal gates should support a claim of recovered structure.
