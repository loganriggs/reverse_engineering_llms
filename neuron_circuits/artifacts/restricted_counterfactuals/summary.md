# Location-restricted MLP counterfactual patching

Each condition filters the same signed gradient × donor-change ranking. Excluded activations remain at their base-run values. Counts are the first tested and integer-refined points where the donor answer becomes the full-vocabulary top-1 token.

| Task | Unrestricted | No layer 23 | No layers 20–23 | No final position | No late layers or final position |
|---|---:|---:|---:|---:|---:|
| IOI | 6.5 (12/12) | 6.5 (12/12) | 6.5 (12/12) | 54.5 (12/12) | 59 (12/12) |
| Agreement | 8.5 (12/12) | 10.5 (12/12) | 21 (12/12) | 19 (12/12) | 19 (12/12) |
| Brackets | 3.5 (12/12) | 4 (12/12) | 6.5 (12/12) | 6.5 (12/12) | 6.5 (12/12) |

All 12 pairs eventually flip in every condition, but some restricted crossings require hundreds of thousands or millions of activations. The success count alone therefore does not establish sparse control.

## Main findings

- Removing only layer 23 has little effect: medians change from 6.5 to 6.5 for IOI, 8.5 to 10.5 for agreement, and 3.5 to 4 for brackets.
- Removing layers 20–23 leaves the IOI median unchanged at 6.5, but raises agreement to 21 and brackets to 6.5. One difficult IOI reverse-direction pair becomes a million-activation outlier.
- Removing the final token position raises IOI from 6.5 to 54.5 activations, agreement from 8.5 to 19, and brackets from 3.5 to 6.5.
- With both late layers and the final position forbidden, the medians are 59 for IOI, 19 for agreement, and 6.5 for brackets. Under a 100-activation budget, 8/12 IOI, 9/12 agreement, and 11/12 bracket pairs still flip.
- Under the strictest restriction, every IOI pair that stays below 1,000 activations uses the repeated-subject position; every sparse agreement pair uses the grammatical-subject position; and every sparse bracket pair uses the unmatched-opener position.

## Interpretation

The original result is not a final-layer artifact. It is partly a final-position steering result: that shortcut is especially important for IOI. Even after removing it, most pairs retain sparse control through earlier token positions, which supports a nontrivial source-position mechanism. The extreme outliers show that this claim is not uniform across prompts or counterfactual directions.
