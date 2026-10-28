# Data/rank red team of the Layer-8 state router

State rank and ridge strength are selected by query-grouped FineWeb cross-validation, then frozen for WikiText. Both FineWeb sites per query are used only after model selection for the final OOD fit.

## Locked hypotheses

- **D0_previous_failure_was_data_limited: not supported.**
- **D1_linear_state_router_validates: not supported.**
- **D2_linear_state_address_plateaus: supported.**

Selected rank `8` and ridge alpha `10.0`.

| Split | Median fidelity | 10th percentile | Minimum |
|---|---:|---:|---:|
| fineweb_train | 0.991 | 0.820 | 0.570 |
| fineweb_cv | 0.990 | 0.696 | -1.558 |
| wikitext | 0.983 | 0.793 | -1.942 |

The original 16-site WikiText 10th percentile was 0.735. A plateau here rejects only this linear low-rank state router; nonlinear routers or additional state variables remain live.
