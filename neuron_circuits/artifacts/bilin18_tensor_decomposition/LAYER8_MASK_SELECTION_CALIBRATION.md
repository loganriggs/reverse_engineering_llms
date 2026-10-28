# Calibration of suffix-mask selection regret

Pairwise held-out mask selection is compared with the exact expected regret of choosing a held-out mask uniformly at random. Surface flatness is the fraction of all 84 masks within 0.01 fidelity of the sitewise best.

## Locked hypotheses

- **F0_broad_flat_top: not supported.**
- **F1_pairwise_has_selection_skill: supported.**
- **F2_low_regret_is_only_flatness: not supported.**

| Corpus | Near-optimal fraction median/q10 | Pairwise regret median/q90 | Random regret median/q90 |
|---|---:|---:|---:|
| fineweb | 0.107/0.013 | 0.002/0.107 | 0.308/2.145 |
| wikitext | 0.125/0.013 | 0.002/0.022 | 0.035/1.148 |

Low regret only counts as routing information if it materially beats the random-choice baseline. This calibration does not rescue the pairwise model's modest held-out R2.
