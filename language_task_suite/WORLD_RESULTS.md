# Document-world and attention-bond pilot (2026-11-02)

## Why this version exists

The first 12-task suite reached high aggregate accuracy but repeated about 90% of evaluation prefixes. This version samples a latent document world: three objects have owners, colors, and locations; a variable sequence of transfers, recolorings, and moves updates those facts; the model must complete a final owner/color/location statement. Each world is rendered as prose, notes, or a summary. The label is recomputed by replaying events, independently of the wording. This gives explicit causal variables and variable-length contexts rather than a handful of fixed prefixes.

The data generator and verifier are `document_world.py` and `validate_world.py`. Across 32,000 audited examples, every target matches replay, every causal edit changes the target, every nuisance edit preserves it, and none of the three evaluation splits contains an exact training prefix. The splits are: fresh worlds with seen object–answer combinations; held-out object–answer combinations selected by a stable hash; and an unseen summary realization using **only vocabulary present in training**. This last check removes the unseen-word confound of the first discarded pilot.

## The model has not learned the general rule

Four 48-dimensional bilinear models with 1–4 attention+MLP blocks, RMSNorm, and 5,000 answer-only updates were trained on 20,000 worlds. All use one seed. Full-vocabulary answer accuracy:

| Blocks | Fresh worlds | Held-out combinations | Unseen summary style |
|---:|---:|---:|---:|
| 1 | 62.9% | 26.9% | 41.9% |
| 2 | 69.8% | 30.1% | 32.1% |
| 3 | 72.4% | 21.2% | 43.6% |
| 4 | 72.3% | 15.8% | 38.9% |

The composition result is the main finding. Larger depth did not monotonically help; the four-block model is around chance on held-out object–answer pairings (chance depends on whether the answer is one of 10 names or six colors/places). This is compatible with learning familiar bindings or lexical associations without a fully reusable state-update algorithm. It does **not** prove the exact shortcut used. In 300 paired edits per family, the three-block model's causal pair accuracy on fresh worlds is 49–58%, versus 12–27% on held-out combinations. On the held-out combinations its prediction remains unchanged after a causal edit in roughly 40–50% of pairs. Nuisance edits are more often ignored, but success on those is not enough to claim correct binding.

The split is deliberately stringent: each withheld object–answer pair occurs **over a thousand times in training contexts** as an initial fact or event, but never as a supervised final answer for that object. A three-block model predicts a withheld pair on only 0.4–2.5% of fresh in-distribution cases. On the composition split it emits a withheld pair on about 17–31% of cases, approximately its task accuracy there. This suggests a learned output prior against withheld combinations, which can override information present in the document. It does not distinguish whether an internal state tracker exists but loses at readout. `analyze_world_bias.py` records this check.

`train_world.py`, `world_v1.json`, the four checkpoints, and `eval_world_counterfactual.py` reproduce these measurements. A capacity follow-up trained a 96-dimensional, eight-head, four-block model on 100,000 worlds for 8,000 updates. It reaches **97.5% on fresh worlds, 42.7% on held-out object–answer combinations, and 47.6% on the unseen summary style** (`world_capacity_block4.json`). Thus poor combination/style generalization persists even when the same architecture nearly solves the fresh-world task. These are one-seed results and the splits have different answer-family mixes; family-level scores are in the JSON.

On 300 paired edits per family, the capacity model gives both correct answers on roughly 90–92% of fresh-world causal pairs, and stays correct under a nuisance edit on roughly 97–98%. On held-out combinations, both-correct causal pair accuracy drops to 26% for color, 52% for place, and 58% for owner (`world_capacity_pairs.json`). A ridge readout fitted to ordinary training-combination residuals reaches 97.3% on fresh worlds but only 42.1% on held-out combinations at the final layer. A diagnostic readout fitted **using held-out-combination labels** reaches 80.5% on separate held-out-combination examples. That suggests considerable answer information is linearly accessible in the final residual *once the withheld mappings are supervised*, but does not show the model has a reusable causal state representation or solve the blind generalization problem (`world_capacity_residual_probe.json`).

## Exact bond gauges and sparse gates

On the final attention bond of the three- and four-block models, `world_bond_gauges.py` fits exact orthogonal coordinates using physical axes, activation/weight PCA, activation/weight varimax, a causal-edit varimax, and random rotations. The fit sees owner and color worlds from the training composition fold; **place worlds are wholly withheld from featurizer fitting**, although the model itself was trained on all three tasks. Full-rank rotations reproduce the original logits to numerical precision. A top-|coordinate| gate then creates a lossy, sample-dependent sparse approximation at the answer position. Evaluation separates task accuracy, agreement with the native model, bond reconstruction error, logit error, and KL on fresh worlds, held-out combinations, and the unseen style.

At eight retained coordinates in the three-block model, physical and activation-varimax bases both score about 73% on fresh worlds. Activation varimax halves relative bond MSE (0.062 versus 0.125); activation PCA gets 71.3% with MSE 0.104, and weight varimax gets 70.3% with MSE 0.413. A post-fit, equal-family causal-edit-energy purity score is 0.955 for activation varimax, 0.873 for physical coordinates, and 0.796 for activation PCA. In the four-block model, activation varimax and activation PCA score 76% at eight coordinates versus 72.7% for physical coordinates on this 300-example gauge test; post-fit purity is 0.891 versus physical 0.856. These are **single-model, small-sample exploratory results**. The physical basis is strong, and improved purity or reconstruction alone is not evidence that an axis describes a human-readable computation. Held-out combination accuracy stays poor under every gauge because the native model is poor there. No sparse basis is credited with recovering the target algorithm.

The label-free activation varimax and the edit-supervised varimax must be interpreted separately. In the latter, causal edits for owner/color are allowed during fitting; neither sees place labels or edits. The post-fit signature evaluation uses all three families and cannot be used to choose a basis without invalidating the holdout.

The larger competent model provides a stronger negative control: at 16 retained final-bond coordinates, physical axes, activation varimax, and edit varimax all preserve roughly 98–99% fresh-world accuracy. Activation varimax reduces relative bond reconstruction MSE from 0.033 to 0.023, but post-fit causal-energy purity is 0.976, slightly below the physical basis's 0.978. All sparse bases leave held-out-combination accuracy near the native model's level. Thus a rotation can help compress this bond, but the current experiment does **not** show a more human-legible or more generalizable causal decomposition (`world_capacity_bond_gauges.json`).

## Realism check

`compare_realism.py` compares crude surface statistics with the frozen 480-window FineWeb panel. FineWeb sentences average about 17.8 words in the regex audit; the generated prose, notes, and summaries average roughly 7.1, 7.2, and 6.2. The synthetic sample uses only about 40–47 distinct alphabetic words per style across 1,000 examples. A 513-token FineWeb window is not a complete document, so document-length and type/token ratios are not directly comparable; the sentence-length and lexical-diversity gaps are nevertheless large. This dataset is a controlled causal benchmark, **not a verified match to web language**. The next generator should use longer, more varied sentences and background discourse, then be checked against a manually annotated natural-text sample before claims of realism.

There is also a small surface bug in this frozen v1 generator: it can write `a orange ...` instead of `an orange ...`. The older 12-family suite has other awkward phrases. These do not change the latent oracle but further limit linguistic realism. The v1 checkpoint and results should remain tied to this exact generator; correct the article in a versioned next generator and retrain rather than silently change the source of these results.

## Next decisions

1. Add a manually reviewed, longer-document v2 with grammatical realization, paraphrases, irrelevant clauses, and background discourse. Quantify its surface statistics and its causal oracle before training.
2. Separate the color composition failure from output-label suppression by testing neutral answer heads, balanced final-answer sampling, and causal swaps at several residual sites. Keep the held-out pairings untouched during all model and featurizer fitting.
3. On a model that generalizes across combinations, compare label-free and intervention-guided sparse gauges against the physical basis, measuring coordinate stability across seeds and whether axes predict causal edit families on a wholly held-out task. Bond MSE and ordinary accuracy alone are insufficient.
