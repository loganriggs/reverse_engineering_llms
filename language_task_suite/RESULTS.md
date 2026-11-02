# First language-suite results (2026-11-01)

These are **pilots**, not evidence that a model has learned natural language. The exact grammar, training code, JSON results, and checkpoints are in this folder. All reported models use the original bilinear `DeepModel` with 48-dimensional residual stream, four heads, one bilinear-attention and one bilinear-MLP module per block, and RMSNorm. RMSNorm breaks the globally polynomial/tensor interpretation, while preserving bilinear modules for later local analysis. Each run uses one seed. The default objective supervises the final answer token; the full-LM run supervises all next tokens of each generated sequence.

| Training run | Blocks | Answer accuracy, variant 0 | Answer accuracy, variant 1 |
|---|---:|---:|---:|
| 12 tasks, both variants, answer-only, 3k steps | 1 | 91.5% | 91.9% |
| same | 2 | 92.5% | 91.8% |
| same | 3 | 92.2% | 92.2% |
| same | 4 | 91.9% | 91.3% |
| 12 tasks, both variants, full next-token LM, 3k steps | 3 | 91.5% | 93.6% |

**These are largely fit scores.** With both variants in training, 90.2% of variant-0 and 89.7% of variant-1 test prefixes exactly occur in the training set. The finite templates are small. With variant 1 withheld entirely, exact overlap is zero, and the earlier 1–4-block answer-only pilot obtained 59–68% there while getting 92–95% on variant 0. The gain from seeing both variants therefore cannot be described as systematic compositional generalization. A simple last-mentioned-candidate baseline gets 35.3% on variant 0 and 37.7% on variant 1 under the corrected generator; the per-family gap is more informative than the aggregate.

The most useful family-level contrast in the mixed answer-only run is that three-block state update, correction with an intervening other object, and active/passive role queries score 100% on both variants, while attribute binding scores only 56%/55% and conditional-rule selection 51%/52%. The full-LM objective improves conditional-rule accuracy to 70%/72.5% at three blocks, but attribute binding remains 48%/51%. In an isolated two-family training run with both variants, three blocks reached nearly perfect answer accuracy, so the failure in the 12-task mixture is not a broken oracle or intrinsic impossibility for this architecture at this size. It may reflect limited training allocation, task interference, or optimization; this pilot does not separate those.

## Causal edit check

`counterfactual.py` evaluates 200 paired examples per family, variant, and edit type. A **causal edit** changes the queried owner/color, queried conditional branch, or tracked object's new location and thus changes the correct answer. A **nuisance edit** changes an irrelevant other owner's color, nonqueried rule key, or other object's location, keeping the answer fixed. Pair accuracy requires both answers correct. Prediction invariance measures whether both model predictions are equal; it should be low for causal edits and high for nuisance edits.

For the 12-task answer-only, three-block model:

| Family | Causal pair accuracy, variants 0/1 | Nuisance pair accuracy, variants 0/1 | Reading |
|---|---:|---:|---|
| State update | 100% / 100% | 100% / 100% | Follows tracked object's location and ignores unrelated location |
| Attribute binding | 13% / 18% | 30.5% / 26.5% | Does not reliably bind owner to color; irrelevant color can change prediction |
| Conditional rule | 0% / 0% | 48% / 51.5% | Typically ignores which color was queried |

The full-LM three-block model keeps state-update pairs at 100%; conditional causal pair accuracy improves to 34%/59%, but attribute causal pair accuracy is only 2.5%/1%. These edits test answer behavior, **not** internal circuit identity.

A positive-control three-block model trained only on attribute binding and conditional rules reaches 99.9%/99.6% answer accuracy on the two variants. In the same paired-edit test, its attribute causal-pair accuracy is 98.5%/99% and nuisance-pair accuracy is 99.5%/99.5%; conditional causal and nuisance pairs are all 100%. Its predictions change on causal edits and remain fixed on nuisance edits at the expected rates. This supports the edit implementation and shows that the architecture can learn the intended selective binding when those tasks receive enough focused training. It does not yet distinguish whether the 12-task failure is update allocation, capacity sharing, or optimization.

## FineWeb comparison and limitations

The 480-window FineWeb audit finds forms useful for choosing document genres and task families: parentheses and quotations are common, while numbered lists and weekdays are less frequent. Manual review of a few matched contexts showed that surface cues often have unrelated meanings: a weekday often appears in a date, and `put` often does not mark a tracked object moving. Thus this is form calibration, not an estimate of how often our exact tasks occur. The toy sentences are much shorter, cleaner, and lexically narrower than FineWeb; no claim of matched distribution is justified.

The next data-design step is a document-level latent world model with variable-length event chains, genre-specific prose realization, and held-out combinations. Evaluate whole-family transfer and counterfactual behavior before using these checkpoints to test canonicalization or sparsification. A high aggregate task score here can conceal failure on the exact binding mechanisms we most want to understand.
