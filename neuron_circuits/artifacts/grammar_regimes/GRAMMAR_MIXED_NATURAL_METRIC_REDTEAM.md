# Natural-metric red team of the mixed-model regime

The operator family is remeasured after full activation-covariance weighting, softmax-Fisher weighting, and restriction to the current correct-versus-rival margin. These metrics suppress sensitivity directions that natural activation changes or output behavior barely use. High-confidence results use the 120 largest-margin held-out rule predictions.

## Locked hypotheses

- **N0_covariance_gap_survives: supported.**
- **N1_fisher_gap_survives: supported.**
- **N2_margin_gap_survives: supported.**
- **N3_mixed_heldout_matches_mixed_train: not supported.**
- **N4_high_confidence_gap_survives: supported.**
- **N5_depth_compression_survives_natural_metric: not supported.**

| Seed | Layer | Group | RMS | Covariance | Fisher-covariance | Margin-covariance | High-conf Fisher |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0 | heldout_generalized | 73.0 | 65.8 | 31.1 | 20.5 | 34.3 |
| 0 | 0 | train_unselected | 74.7 | 67.3 | 12.3 | 20.8 | 14.6 |
| 0 | 0 | train_selected | 74.5 | 66.9 | 15.1 | 20.4 | 18.2 |
| 0 | 0 | pure_same_heldout | 12.8 | 13.4 | 2.2 | 3.5 | 2.3 |
| 0 | 1 | heldout_generalized | 70.1 | 66.2 | 32.7 | 26.0 | 40.0 |
| 0 | 1 | train_unselected | 71.1 | 66.5 | 13.1 | 26.4 | 14.5 |
| 0 | 1 | train_selected | 72.6 | 68.1 | 16.2 | 26.6 | 17.9 |
| 0 | 1 | pure_same_heldout | 15.9 | 14.8 | 1.8 | 6.1 | 1.3 |
| 0 | 2 | heldout_generalized | 41.6 | 39.8 | 29.7 | 20.5 | 36.7 |
| 0 | 2 | train_unselected | 43.3 | 41.4 | 13.7 | 21.3 | 14.6 |
| 0 | 2 | train_selected | 42.7 | 40.4 | 16.2 | 21.4 | 19.2 |
| 0 | 2 | pure_same_heldout | 12.8 | 11.0 | 2.0 | 5.2 | 1.9 |
| 0 | 3 | heldout_generalized | 13.8 | 12.4 | 13.5 | 6.7 | 14.1 |
| 0 | 3 | train_unselected | 15.5 | 14.1 | 7.4 | 6.7 | 7.5 |
| 0 | 3 | train_selected | 18.1 | 16.4 | 9.0 | 6.7 | 10.0 |
| 0 | 3 | pure_same_heldout | 8.6 | 7.3 | 2.0 | 6.2 | 1.8 |
| 1 | 0 | heldout_generalized | 74.0 | 67.3 | 33.6 | 22.6 | 29.4 |
| 1 | 0 | train_unselected | 74.8 | 68.5 | 16.2 | 22.9 | 18.2 |
| 1 | 0 | train_selected | 75.7 | 69.0 | 19.8 | 22.2 | 26.0 |
| 1 | 0 | pure_same_heldout | 21.9 | 20.3 | 3.6 | 4.8 | 2.1 |
| 1 | 1 | heldout_generalized | 66.0 | 60.7 | 32.7 | 22.7 | 30.7 |
| 1 | 1 | train_unselected | 66.0 | 61.3 | 15.2 | 24.3 | 17.4 |
| 1 | 1 | train_selected | 67.9 | 62.4 | 20.7 | 23.6 | 23.9 |
| 1 | 1 | pure_same_heldout | 19.4 | 16.4 | 3.2 | 5.8 | 1.5 |
| 1 | 2 | heldout_generalized | 39.8 | 38.4 | 31.6 | 19.7 | 29.5 |
| 1 | 2 | train_unselected | 38.8 | 37.1 | 15.8 | 19.3 | 16.7 |
| 1 | 2 | train_selected | 40.1 | 38.3 | 20.1 | 20.0 | 23.1 |
| 1 | 2 | pure_same_heldout | 14.9 | 13.6 | 3.0 | 5.9 | 1.5 |
| 1 | 3 | heldout_generalized | 14.1 | 12.8 | 13.5 | 6.6 | 13.5 |
| 1 | 3 | train_unselected | 16.6 | 15.2 | 9.5 | 6.7 | 9.1 |
| 1 | 3 | train_selected | 15.1 | 13.5 | 10.8 | 6.8 | 11.3 |
| 1 | 3 | pure_same_heldout | 9.0 | 7.6 | 3.4 | 6.1 | 2.4 |
| 2 | 0 | heldout_generalized | 73.4 | 66.5 | 28.0 | 21.6 | 33.8 |
| 2 | 0 | train_unselected | 76.1 | 69.8 | 18.0 | 21.3 | 15.8 |
| 2 | 0 | train_selected | 77.0 | 70.1 | 26.4 | 22.7 | 24.2 |
| 2 | 0 | pure_same_heldout | 27.6 | 21.4 | 2.6 | 5.9 | 1.3 |
| 2 | 1 | heldout_generalized | 70.5 | 65.8 | 29.6 | 24.2 | 34.9 |
| 2 | 1 | train_unselected | 71.0 | 66.4 | 17.2 | 25.0 | 15.7 |
| 2 | 1 | train_selected | 73.6 | 68.3 | 25.9 | 26.8 | 22.4 |
| 2 | 1 | pure_same_heldout | 25.7 | 21.5 | 3.8 | 6.7 | 3.5 |
| 2 | 2 | heldout_generalized | 41.4 | 39.1 | 30.4 | 20.2 | 32.4 |
| 2 | 2 | train_unselected | 41.5 | 39.2 | 18.3 | 21.0 | 16.9 |
| 2 | 2 | train_selected | 41.9 | 39.8 | 23.8 | 21.9 | 22.2 |
| 2 | 2 | pure_same_heldout | 20.2 | 16.6 | 3.9 | 6.2 | 3.8 |
| 2 | 3 | heldout_generalized | 12.0 | 10.8 | 12.5 | 6.6 | 13.5 |
| 2 | 3 | train_unselected | 13.4 | 12.2 | 9.8 | 6.7 | 8.9 |
| 2 | 3 | train_selected | 15.8 | 14.2 | 11.3 | 6.5 | 10.0 |
| 2 | 3 | pure_same_heldout | 9.2 | 8.3 | 3.8 | 5.7 | 4.2 |

A surviving gap is evidence about the causal operator family in the model's used local geometry. It still does not prove that no alternative nonlinear global description can compress the mixed computation.
