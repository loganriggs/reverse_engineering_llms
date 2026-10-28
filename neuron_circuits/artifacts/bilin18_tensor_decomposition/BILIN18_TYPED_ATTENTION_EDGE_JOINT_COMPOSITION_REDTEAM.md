# Red team: joint typed-edge composition

The six-layer closed-loop program is repeated on a new FineWeb split. An all-types program tests exact partition/execution, and fixed random 128-type programs test whether the discovery-selected vocabulary is special.

Maximum native-branch error: `5.34e-05`. Fine-128 absolute mean KL error: `0.1850`.

## Locked hypotheses

- **R0_all_types_reproduce_native: supported.**
- **R1_selected128_beats_random128: supported.**
- **R2_selected128_has_small_absolute_error: supported.**
- **R3_selected128_prediction_agreement_replicates: supported.**
- **R4_fine32_is_not_enough_for_predictions: supported.**
- **native_branch_matches_checkpoint: supported.**

| Method | KL recovery | CE delta | Top-1 agreement |
|---|---:|---:|---:|
| fine32 | 0.5780 | +2.9840 | 0.2549 |
| fine128 | 0.9738 | +0.1741 | 0.7988 |
| random128 | 0.3936 | +4.2707 | 0.1333 |
| all | 1.0000 | -0.0000 | 1.0000 |

## Guardrails

Random supports are sampled independently in each layer from types active on discovery, without matching head, distance, or category marginals. Thus they are a coarse null; the earlier coordinate-preserving controls remain the stronger test of individual coordinates. The all-types branch is the implementation invariant. Normalized recovery is accompanied by absolute KL, CE, and top-1 because the no-attention denominator is intentionally severe.
