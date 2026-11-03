# Weight-only versus training-activation OV bases for two closure rules

The [within-head SVD test](CROSS_SOURCE_OV_MODE_RESULT.md) showed that sports bracket and FineWeb delimiter closure share a necessary late head but are not isolated by one weight-SVD mode. This experiment changes the **importance metric** for truncating that head's (O_hV_h) product. A weight-only SVD is compared with three activation-weighted SVDs fit on sports training reports, FineWeb delimiter training windows, or an equal mixture of their uncentered activation Grams. All bases are fit without evaluation examples. Rank retention is scored on a separate 100-game sports panel and the untouched 164-document FineWeb spelling×delimiter panel.

The basis fit uses 128 distinct sports training games (all four arms, 106,292 non-padding tokens) and 512 sampled delimiter-training windows (123,962 non-padding tokens). The sports and FineWeb fit sources are separate; no document containing the heldout two-rule combination is used. The critical second-attention-layer head is the one localized by complete-head ablation in each seed. At every tested rank, only that head's OV product changes; QK, other heads, MLPs, embeddings, and output weights remain fixed. Full rank reproduces the baseline scores. The [runner](compare_cross_source_ov_bases.py) records rule-site scores, sports alias control, and ordinary token negative log-likelihood (NLL) separately.

| Seed | Basis | Rank 2 sports / FineWeb paired | Rank 4 sports / FineWeb paired | Rank 8 sports / FineWeb paired |
|---|---|---:|---:|---:|
| 41701 | Weight-only | .980 / .421 | .990 / .500 | 1.000 / .530 |
| 41701 | Sports activations | .870 / .335 | .990 / .482 | 1.000 / .518 |
| 41701 | FineWeb activations | .770 / .274 | .990 / .482 | 1.000 / .527 |
| 41701 | Balanced activations | .820 / .351 | .990 / .482 | 1.000 / .524 |
| 41702 | Weight-only | .990 / .387 | .990 / .384 | .990 / .662 |
| 41702 | Sports activations | .080 / .000 | .710 / .201 | .990 / .665 |
| 41702 | FineWeb activations | .100 / .018 | .920 / .515 | .990 / .668 |
| 41702 | Balanced activations | .080 / .006 | .850 / .366 | .990 / .665 |

The full-rank reference is sports bracket 1.000/.990 and FineWeb delimiter .527/.683 in seeds 41701/41702. **Weight-only is the strongest low-rank joint preservation baseline here.** At rank 2 it preserves nearly all sports bracket behavior in both seeds, whereas activation weighting substantially degrades seed 41702. At rank 4, FineWeb-activation weighting improves seed 41702's FineWeb delimiter score (.515 versus .384 weight-only) but lowers its sports score (.920 versus .990). At rank 8, all four bases retain nearly all sports behavior and similar FineWeb behavior. This is a tradeoff, not evidence that one data metric identifies a more legible circuit.

Token NLL does not resolve the rule question. For example, seed 41702's weight-only rank 4 has FineWeb token NLL **7.776**, lower than the full-rank **7.890**, yet its delimiter paired score falls from .683 to .384. Truncation can improve background calibration or shift error among tokens while breaking the controlled rule. Sports alias paired accuracy remains within a few points of baseline for all tested bases and ranks; the closure changes are relatively rule-specific. The complete numbers, including rank 1 and full rank, are in the [aggregate JSON](results/cross_source_ov_basis_comparison.json).

This comparison does not test an explicitly **causal-rule-optimized** basis. Activation energy weights frequently visited input directions, which need not be the directions supporting a rare rule. The next method should fit a rank-constrained OV replacement on *training-only* causal pairs, balancing teacher fidelity, closure responses on both sources, and unrelated-token/alias preservation; then use whole-task heldouts to compare its frontier with the weight-only and activation baselines. Any method that fits labels is supervised and must be reported separately from label-free discovery. The larger sampled-language DGP and its task-window realism remain the primary research objective; these empirical carriers are a method test, not a substitute for that generator.
