# Weight-only outlier test for the L17H2 edit compressor

Every c_v head slice in all 18 layers is scored by top singular value divided by RMS singular value. The L17H2 left singular basis is then used without activation fitting to reconstruct post-boundary value changes and finite spelling effects, including new lexemes.

## Locked results

- **W0_H2_has_a_spectral_outlier: supported.**
- **W1_H2_is_prospectively_ranked: not supported.**
- **W2_weight_rank1_reconstructs_value_changes: supported.**
- **W3_weight_rank1_is_causally_sufficient: supported.**
- **W4_weight_rank1_generalizes_new_lexemes: supported.**
- **W5_random_rank1_fails: supported.**
- **W6_weight_only_outlier_identifies_the_read_function: not supported.**

L17H2 score: `1.721`; Gaussian rectangular edge: `1.333`; global head rank: `60/162`.

| Case | Fold/split | raw weight-R1 | causal weight-R1 | best random causal R1 |
|---|---|---:|---:|---:|
| original | 0/0 | 0.739 | 0.996 | 0.016 |
| original | 0/1 | 0.690 | 0.996 | 0.014 |
| original | 1/0 | 0.741 | 0.997 | 0.016 |
| original | 1/1 | 0.760 | 0.997 | 0.015 |
| original | 2/0 | 0.671 | 0.998 | 0.018 |
| original | 2/1 | 0.725 | 0.998 | 0.016 |
| new_cues | -/0 | 0.657 | 0.997 | 0.017 |
| new_cues | -/1 | 0.774 | 0.997 | 0.018 |

## Interpretation boundary

Passing the behavioral gates would show that the circuit-relevant value coordinate is directly readable from weights, without activation-derived PCA. Spectral rank determines whether the same outlier would identify this head in a blind whole-model search; behavioral success alone does not make the outlier globally unique or semantic.
