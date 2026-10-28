# Gauge and common-panel red team for grammar Jacobian charts

Every trained regime is evaluated on exactly the same prompts within a seed. The panel is balanced over attribute, structured answer, and arbitrary answer, and every prompt is answered correctly by all three trained regimes in both discovery and held-out templates. RMS and standard-deviation normalization remove each bilinear channel's legal scale gauge.

## Locked hypotheses

- **R0_gauge_normalization_is_invariant: not supported.**
- **R1_common_panel_rms_load_gap: supported.**
- **R2_common_panel_std_load_gap: supported.**
- **R3_common_panel_dictionary_gap: supported.**
- **R4_arbitrary_geometry_is_closer_to_untrained: not supported.**
- **R5_half_arbitrary_is_already_high_rank: supported.**
- **R6_load_contrast_persists_upstream: supported.**

| Seed | Model | Layer | raw rank | RMS rank | std rank | RMS r16 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0 | 20.6 | 16.1 | 16.3 | 0.802 |
| 0 | 0.0 | 1 | 19.2 | 18.4 | 18.3 | 0.873 |
| 0 | 0.0 | 2 | 17.4 | 14.8 | 14.8 | 0.984 |
| 0 | 0.0 | 3 | 9.7 | 9.1 | 9.0 | 0.998 |
| 0 | 0.5 | 0 | 167.6 | 150.6 | 151.2 | 0.192 |
| 0 | 0.5 | 1 | 142.6 | 140.8 | 140.9 | 0.293 |
| 0 | 0.5 | 2 | 59.4 | 59.1 | 59.1 | 0.604 |
| 0 | 0.5 | 3 | 18.1 | 18.2 | 18.1 | 0.984 |
| 0 | 1.0 | 0 | 155.2 | 142.5 | 144.5 | 0.183 |
| 0 | 1.0 | 1 | 132.3 | 130.3 | 130.4 | 0.304 |
| 0 | 1.0 | 2 | 48.8 | 48.3 | 48.4 | 0.688 |
| 0 | 1.0 | 3 | 17.1 | 16.7 | 16.7 | 0.986 |
| 0 | untrained | 0 | 11.0 | 10.4 | 10.9 | 1.000 |
| 0 | untrained | 1 | 10.8 | 10.2 | 10.7 | 1.000 |
| 0 | untrained | 2 | 9.6 | 9.3 | 9.5 | 1.000 |
| 0 | untrained | 3 | 6.8 | 6.5 | 6.9 | 1.000 |
| 1 | 0.0 | 0 | 34.0 | 23.4 | 23.2 | 0.749 |
| 1 | 0.0 | 1 | 27.2 | 25.0 | 25.0 | 0.875 |
| 1 | 0.0 | 2 | 22.9 | 17.5 | 17.6 | 0.931 |
| 1 | 0.0 | 3 | 10.7 | 8.3 | 8.3 | 0.991 |
| 1 | 0.5 | 0 | 167.4 | 159.7 | 159.7 | 0.184 |
| 1 | 0.5 | 1 | 128.3 | 126.9 | 126.9 | 0.336 |
| 1 | 0.5 | 2 | 56.6 | 56.3 | 56.3 | 0.645 |
| 1 | 0.5 | 3 | 18.8 | 18.8 | 18.8 | 0.982 |
| 1 | 1.0 | 0 | 144.1 | 124.7 | 125.9 | 0.213 |
| 1 | 1.0 | 1 | 136.7 | 134.1 | 134.4 | 0.254 |
| 1 | 1.0 | 2 | 52.8 | 52.4 | 52.5 | 0.632 |
| 1 | 1.0 | 3 | 14.3 | 13.5 | 13.6 | 0.984 |
| 1 | untrained | 0 | 10.0 | 9.3 | 9.8 | 1.000 |
| 1 | untrained | 1 | 9.2 | 8.6 | 8.7 | 1.000 |
| 1 | untrained | 2 | 8.8 | 8.8 | 8.7 | 1.000 |
| 1 | untrained | 3 | 5.3 | 5.3 | 5.2 | 1.000 |

## Explicit gauge test

The gauge-transformed model's maximum logit difference was 5.53e-05. Its maximum effective-rank change was 4.604 raw, 7.04e-04 after RMS normalization, and 9.54e-05 after standard-deviation normalization.

The untrained comparison distinguishes arbitrary memorization from generic high-dimensional random computation. It is a diagnostic baseline, not a matched-performance model.
