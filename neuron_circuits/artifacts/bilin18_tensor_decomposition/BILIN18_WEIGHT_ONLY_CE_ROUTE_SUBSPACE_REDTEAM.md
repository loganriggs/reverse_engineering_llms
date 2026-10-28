# Reconstruction red team for weight-only CE-route bases

The original half-of-CE fidelity gate can pass by dimensionality alone: four directions inhabit a nine-dimensional head space. This audit evaluates the discovery-selected weight view against the best of four random rank-four bases on the same replication routes, and against the earlier empirical natural-route basis.

## Locked hypotheses

- **R0_material_weight_reconstruction_beats_dimensional_null: not supported.** Observed: `{'layers': [4, 9, 13, 14, 15, 17]}`
- **R1_weight_geometry_beats_empirical_generic_routes: not supported.** Observed: `{'layers': [1, 4]}`
- **R2_early_weight_signal_is_selective: supported.** Observed: `{'layers': [1, 4]}`
- **R3_half_ce_threshold_was_too_weak: supported.** Observed: `{'original_passed_all': True, 'null_passing_layers': 6}`

| L | Weight view | Weight fidelity | Best-4 random mean/p95/pct | Generic fidelity | Weight−generic |
|---:|---|---:|---:|---:|---:|
| 0 | ov | 0.576 | 0.524/0.608/0.839 | 0.701 | -0.125 |
| 1 | ov | 0.560 | 0.506/0.574/0.903 | 0.438 | +0.122 |
| 2 | ov | 0.579 | 0.509/0.580/0.948 | 0.576 | +0.003 |
| 3 | scaled_combined | 0.558 | 0.500/0.560/0.944 | 0.631 | -0.073 |
| 4 | qk | 0.584 | 0.495/0.552/0.994 | 0.467 | +0.117 |
| 5 | ov | 0.598 | 0.525/0.611/0.910 | 0.594 | +0.004 |
| 6 | combined | 0.528 | 0.501/0.560/0.776 | 0.610 | -0.081 |
| 7 | combined | 0.561 | 0.525/0.616/0.742 | 0.721 | -0.160 |
| 8 | combined | 0.519 | 0.500/0.566/0.697 | 0.691 | -0.173 |
| 9 | scaled_combined | 0.581 | 0.508/0.578/0.955 | 0.676 | -0.095 |
| 10 | scaled_combined | 0.608 | 0.535/0.627/0.887 | 0.736 | -0.128 |
| 11 | scaled_combined | 0.574 | 0.524/0.602/0.839 | 0.674 | -0.101 |
| 12 | combined | 0.521 | 0.488/0.540/0.882 | 0.612 | -0.091 |
| 13 | combined | 0.552 | 0.492/0.545/0.972 | 0.586 | -0.034 |
| 14 | scaled_combined | 0.672 | 0.549/0.651/0.983 | 0.694 | -0.022 |
| 15 | scaled_combined | 0.683 | 0.507/0.579/1.000 | 0.669 | +0.015 |
| 16 | qk | 0.598 | 0.522/0.607/0.932 | 0.751 | -0.153 |
| 17 | scaled_combined | 0.648 | 0.541/0.644/0.959 | 0.721 | -0.073 |

## Interpretation

A stable static basis can still be a valid description of the checkpoint's operator inventory even when it fails to predict which coordinates FineWeb and next-token loss execute. Passing the empirical-route comparison only in selected early layers would support a hybrid ontology: weights delimit reusable operations, while context/frequency selects their natural coordinate chart.
