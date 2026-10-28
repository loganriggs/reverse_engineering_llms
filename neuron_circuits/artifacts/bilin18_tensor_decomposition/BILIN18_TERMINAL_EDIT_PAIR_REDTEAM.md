# Pairwise red team of the pooled unrelated-edit H6 result

The sixteen substitutions used in the pooled unrelated panel are now analyzed separately. Each split-0 head choice is frozen before evaluation on split 1. The previously pooled H6 choice is also evaluated without reselection on every pair.

## Locked results

- **P0_pair_panels_are_balanced_and_powered: supported.**
- **P1_pooled_H6_recurs_across_pairs: not supported.** (8/16 pairs)
- **P2_H6_effect_is_directionally_consistent: supported.** (14/16 pairs)
- **P3_H6_is_the_common_context_only_selection: not supported.** (4/16 pairs)
- **P4_pair_specific_context_selection_transfers: not supported.** (7/16 pairs)
- **P5_pair_specific_quotient_selection_transfers: not supported.** (4/16 pairs)
- **P6_pooled_H6_is_not_an_averaging_artifact: not supported.**

| Pair | N d/e | context H/rank | quotient H/rank | H6 rank/effect | held-out winner |
|---|---:|---:|---:|---:|---:|
| dog↔cat | 32/32 | H0/2 | H6/6 | 6/+2.84e-07 | H2 |
| man↔woman | 32/32 | H1/4 | H8/1 | 2/+7.37e-06 | H8 |
| red↔blue | 32/32 | H6/5 | H6/5 | 5/+1.01e-06 | H5 |
| Monday↔Tuesday | 32/32 | H1/5 | H1/5 | 7/+9.03e-08 | H2 |
| France↔Germany | 32/32 | H2/9 | H6/3 | 3/+1.88e-07 | H4 |
| Apple↔Microsoft | 32/32 | H1/2 | H1/2 | 1/+4.91e-05 | H6 |
| football↔baseball | 24/32 | H0/5 | H2/1 | 4/+3.04e-06 | H2 |
| summer↔winter | 32/32 | H2/2 | H2/2 | 6/+3.25e-07 | H0 |
| morning↔evening | 32/32 | H2/6 | H2/6 | 5/+5.41e-07 | H1 |
| good↔bad | 32/32 | H1/3 | H5/2 | 1/+1.73e-06 | H6 |
| yes↔no | 32/32 | H6/1 | H6/1 | 1/+6.09e-06 | H6 |
| left↔right | 32/32 | H2/7 | H2/7 | 5/-1.22e-07 | H3 |
| large↔small | 32/32 | H6/4 | H1/9 | 4/-6.03e-08 | H2 |
| hot↔cold | 32/32 | H1/2 | H0/8 | 3/+6.87e-07 | H4 |
| first↔last | 32/32 | H6/1 | H6/1 | 1/+2.23e-06 | H6 |
| black↔white | 32/32 | H2/5 | H1/3 | 1/+1.4e-05 | H6 |

## Interpretation boundary

Pairwise stability would support H6 as a reusable terminal operation. Failure would show that pooling creates a useful average predictor but erases conditional mechanism diversity. Context-only selection tests whether occupied tensor magnitude is enough; quotient selection tests whether the model's changed output distribution is required. Neither assigns a human semantic name to a head.
