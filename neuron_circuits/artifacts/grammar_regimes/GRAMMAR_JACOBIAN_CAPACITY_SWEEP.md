# Fixed-task capacity sweep of local operator families

All widths solve the same 50%-arbitrary task and are evaluated on the same balanced, jointly correct prompts within each data seed. Jacobian columns are RMS-normalized to remove the bilinear channel scale gauge.

## Locked hypotheses

- **C0_manual_trace_is_exact: supported.**
- **C1_capacity_increases_operator_diversity: not supported.**
- **C2_capacity_reduces_fractional_rank: supported.**
- **C3_late_answer_interface_remains_compact: supported.**
- **C4_exceptions_use_more_diverse_operators: not supported.**
- **C5_rule_exception_subspaces_separate_with_width: not supported.**
- **C6_depth_compresses_operator_families: supported.**

| Width | Layer | Effective rank | Modes for 90% | Selected/unselected rank | Excess subspace overlap | r16 fidelity |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0 | 122.2 | 109.0 | 80.7/79.4 | 0.149 | 0.148 |
| 64 | 1 | 117.4 | 108.7 | 78.4/77.4 | 0.227 | 0.212 |
| 64 | 2 | 56.3 | 54.3 | 45.9/45.4 | 0.580 | 0.553 |
| 64 | 3 | 18.1 | 22.3 | 16.8/17.5 | 0.677 | 0.981 |
| 96 | 0 | 119.9 | 107.7 | 80.7/77.1 | 0.159 | 0.163 |
| 96 | 1 | 108.7 | 104.0 | 75.5/72.4 | 0.317 | 0.279 |
| 96 | 2 | 53.1 | 52.3 | 44.4/42.4 | 0.628 | 0.606 |
| 96 | 3 | 18.3 | 22.3 | 17.5/17.3 | 0.649 | 0.984 |
| 192 | 0 | 127.6 | 116.3 | 82.0/82.4 | 0.183 | 0.177 |
| 192 | 1 | 107.7 | 107.3 | 73.1/73.0 | 0.359 | 0.303 |
| 192 | 2 | 57.6 | 59.0 | 45.8/46.1 | 0.608 | 0.567 |
| 192 | 3 | 17.5 | 23.0 | 16.7/16.4 | 0.656 | 0.985 |
| 384 | 0 | 134.0 | 122.7 | 83.5/83.4 | 0.133 | 0.128 |
| 384 | 1 | 115.8 | 110.7 | 76.7/76.7 | 0.287 | 0.247 |
| 384 | 2 | 52.5 | 50.0 | 43.1/42.9 | 0.610 | 0.581 |
| 384 | 3 | 18.6 | 24.7 | 17.5/17.6 | 0.635 | 0.978 |

Absolute rank asks how many distinct local operator modes the model uses. Rank divided by available channels asks how densely it uses capacity. Selected/unselected subspace overlap asks whether exception and rule computations separate rather than forcing them onto one scalar continuum.
