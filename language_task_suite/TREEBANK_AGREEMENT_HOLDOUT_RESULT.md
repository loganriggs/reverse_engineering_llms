# Corpus-derived agreement: whole-subject holdout

This pilot tests whether a variable, parsed-corpus clause generator can provide a rule-known language task for later weight-based interpretability. It is a **single-sentence syntax arm**, not a realistic document DGP or a qualified mechanistic target yet.

## Generator and evaluation

From a public BookCorpus treebank, collect short noun-phrase subjects with a conservative singular/plural head rule and finite verb-phrase continuations whose first word is an inflection of **be**, **have**, or **do**. Independently sample a subject, a verb lemma, and a VP continuation, then choose the verb inflection from the subject's number. An *attractor* subject contains an opposite-number noun in a prepositional phrase, such as a plural subject followed by “of the report” before the finite verb. This creates a possible nearest-noun shortcut. The visible subject phrase determines number in the generator's unambiguous phrase lexicon. Sampled VP lemmas are latent until the verb appears, so evaluation compares only the two inflections of the same lemma; exact top-token accuracy is not an agreement score.

The initial test reused full subject phrases. Its depth-2/depth-4 attractor pair accuracies were .856/.870, but 94% of test attractor phrases appeared in training. That is a memorization confound. We now assign each normalized **whole subject phrase** to train or heldout by SHA-256, with zero phrase overlap. Head nouns and word pieces may still recur. Training uses next-subword cross-entropy at every nonpadding position. The balanced test is 50% plain and 50% attractor, with singular/plural balanced inside each condition. We compare 1-, 2-, and 4-block bilinear transformers with RMS normalization, width 64, four heads, and BPE vocabulary 4,000. The source treebank, code, and test split do not establish a full next-token oracle for all clause words; the agreement site is the rule-known target.

| Fit trees / train clauses / steps | Train attractor rate | Depth | Plain heldout | Attractor heldout | Overall |
|---|---:|---:|---:|---:|---:|
| 20k / 12k / 1500 | .2 | 1 | .832 | .238 | .535 |
| 20k / 12k / 1500 | .2 | 2 | .864 | .330 | .597 |
| 20k / 12k / 1500 | .2 | 4 | .896 | .356 | .626 |
| 20k / 12k / 1500 | .5 | 2 | .824 | .448 | .636 |
| 20k / 12k / 1500 | .5 | 4 | .870 | .430 | .650 |
| 100k / 50k / 3000 | .5 | 2 | .853 | .638 | .746 |
| 100k / 50k / 3000 | .5 | 4 | .908 | .639 | .774 |

Chance pair accuracy is .5. The 20k split had 0 overlapping sampled whole subjects across 6,186 train and 580 test unique phrases; the 100k split had 0 across 18,751 and 1,439. Test BPE unknown-token share was zero in both runs. More and more varied data moved attractor accuracy from below chance to ~.64, but plain clauses remain much easier.

## Paired causal check

Hold the sampled VP continuation and verb lemma fixed. Replace the subject with another heldout phrase of the **same** number or the **opposite** number, matched on attractor presence. In the 100k-fit run, 1,000 baseline clauses were split evenly between plain and attractor. A valid agreement rule should preserve the preferred inflection under same-number replacement and flip it under opposite-number replacement.

| Depth | Group | Baseline correct | Same-number invariant | Opposite-number flip | Both correct responses |
|---:|---|---:|---:|---:|---:|
| 2 | Plain (500) | .856 | .750 | .766 | .642 |
| 2 | Attractor (500) | .660 | .562 | .480 | .290 |
| 4 | Plain (500) | .904 | .818 | .820 | .730 |
| 4 | Attractor (500) | .626 | .550 | .480 | .278 |

Among the 381 attractor cases whose first determiner did not itself signal number, baseline accuracy is only .596 (depth 2) or .559 (depth 4), and both-response causal scores are .283/.270. This is evidence that accuracy on the full attractor panel still relies partly on surface cues and that the models do not robustly implement the intended head-noun rule. It does not identify the exact alternative algorithm; a nearest-noun hypothesis needs its own controlled edit or position-balanced contrast.

## Decision

Keep this generator and the whole-phrase split as a **negative/partial-learning control** for interpretability methods. Do not claim a recovered syntax circuit or score weight decompositions on the attractor rule as though it were reliably learned. The next useful experiments are a broader, discourse-bearing sampled DGP with multiple rule families and controlled form priors, plus a smaller targeted agreement study that orthogonally varies head-number cues, attractor number, distance, and determiner cues. A model may become a weight-method target only after it passes heldout behavior and paired causal edits. Preserve whole-task/combination holdouts and report token reconstruction, site prediction, causal fidelity, and new-task discovery separately.

Reproduction scripts: `treebank_agreement_dgp.py`, `audit_treebank_agreement_dgp.py`, `train_treebank_agreement.py`, and `eval_treebank_agreement_causal.py`. Aggregate metrics are in `results/treebank_agreement_subject_holdout*.json`; model checkpoints and the raw treebank remain local.
