# Interaction-order and gain red-team of the late-MLP path program

The symmetry-invariant program is split into the discovery-free degree≤1 expansion and six degree-two layer interactions. Separately, every path with `d` bilinear layers is rescaled by `gain^d`; this is exactly the path expansion of a hypothetical suffix whose bilinear derivative gain is multiplied while native operands and directions are held fixed.

## Locked hypotheses

- **G0_pair_interactions_are_material: supported.**
- **G1_pair_identity_beats_random_median: not supported.**
- **G2_pair_identity_beats_random_p95: not supported.**
- **G3_pair_ranking_transfers: supported.**
- **G4_half_gain_is_first_order_compact: supported.**
- **G5_native_gain_needs_more_than_first_order: not supported.**
- **G6_high_gain_broadens_the_program: supported.**
- **G7_effective_support_grows_with_gain: supported.**

Discovery pair order: `BBII`, `IBIB`, `BIIB`, `IBBI`, `BIBI`, `IIBB`.

| Corpus | Degree≤1 | + top two pairs | Gain | Random-pair median / p95 | Top-2 overlap |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.880 | 0.933 | 0.053 | 0.924 / 0.931 | 2 |
| wikitext | 0.912 | 0.950 | 0.038 | 0.945 / 0.950 | 2 |
| tinystories | 0.911 | 0.950 | 0.039 | 0.945 / 0.950 | 2 |

| Corpus | Gain | K90 | Effective support | K5 / K8 fidelity |
|---|---:|---:|---:|---:|
| fineweb | 0.5 | 1 | 2.73 | 0.991 / 0.997 |
| fineweb | 1.0 | 5 | 7.02 | 0.902 / 0.949 |
| fineweb | 1.5 | 12 | 12.69 | 0.694 / 0.808 |
| fineweb | 2.0 | 13 | 14.68 | 0.511 / 0.693 |
| wikitext | 0.5 | 1 | 2.55 | 0.993 / 0.998 |
| wikitext | 1.0 | 4 | 6.22 | 0.920 / 0.961 |
| wikitext | 1.5 | 11 | 11.56 | 0.748 / 0.854 |
| wikitext | 2.0 | 12 | 14.58 | 0.567 / 0.725 |
| tinystories | 0.5 | 1 | 2.55 | 0.993 / 0.998 |
| tinystories | 1.0 | 4 | 6.26 | 0.922 / 0.961 |
| tinystories | 1.5 | 11 | 11.65 | 0.750 / 0.853 |
| tinystories | 2.0 | 13 | 14.60 | 0.557 / 0.715 |

The gain sweep is a local counterfactual on the derivative program, not a runnable reparameterized language model. It diagnoses why path compactness occurs: low bilinear gain suppresses high-order interactions; larger gain makes the same contextual operator chain increasingly distributed.
