# Are correctly generalized rules compact inside a mixed model?

The mixed model is tested on unrandomized held-out entity pairs it answers correctly. Those examples were never trained. Their RMS-normalized local operator ranks are compared with exactly relation/answer-matched trained facts, and with the pure-rule model on the identical held-out prompts. All rank estimates use 120 sites and 20 matched resamples.

## Locked hypotheses

- **L0_manual_trace_is_exact: supported.**
- **L1_trained_selected_and_unselected_share_regime: supported.**
- **L2_compact_rule_coexistence: not supported.**
- **L3_global_mixed_regime: supported.**
- **L4_same_prompt_pure_model_is_more_compact: supported.**
- **L5_late_interfaces_are_compact: supported.**
- **L6_mixed_generalization_compresses_with_depth: supported.**

| Seed | Layer | Mixed held-out rule | Mixed trained unselected | Mixed trained exception | Pure model, same prompts |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 73.9±1.5 | 74.4±0.9 | 74.2±1.0 | 13.4±1.5 |
| 0 | 1 | 70.7±1.0 | 70.8±0.8 | 72.3±0.8 | 15.5±1.4 |
| 0 | 2 | 41.9±0.4 | 42.8±0.5 | 43.4±0.8 | 12.8±0.5 |
| 0 | 3 | 14.6±1.4 | 15.6±1.0 | 17.5±0.8 | 8.7±0.2 |
| 1 | 0 | 73.9±1.2 | 71.5±1.9 | 76.7±0.8 | 21.4±1.6 |
| 1 | 1 | 66.0±0.9 | 64.5±1.1 | 67.1±0.6 | 19.5±1.6 |
| 1 | 2 | 39.7±0.5 | 39.3±0.5 | 40.0±0.4 | 14.8±1.0 |
| 1 | 3 | 14.2±1.3 | 16.1±0.9 | 14.5±1.1 | 8.7±0.8 |
| 2 | 0 | 73.7±2.0 | 76.3±0.7 | 77.8±0.5 | 28.3±2.6 |
| 2 | 1 | 70.6±0.6 | 72.5±0.6 | 73.3±0.6 | 26.1±0.4 |
| 2 | 2 | 41.4±0.5 | 42.4±0.5 | 42.2±0.4 | 20.2±0.3 |
| 2 | 3 | 12.2±0.9 | 15.8±0.8 | 14.9±0.9 | 9.2±0.5 |

Correct held-out behavior is necessary for calling an example rule generalization, but it does not force the internal implementation to match the pure-rule model. A high-rank mixed-model chart on these examples would therefore indicate a global regime change, not merely a separate exception lookup table.
