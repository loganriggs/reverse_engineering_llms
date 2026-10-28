# Frozen transfer of directional derivative order

Derivative order is treated as part of the interpreter description. Within FineWeb, each lexical pair is evaluated using the order selected on every other pair. For OOD evaluation, one order per layer is selected on all FineWeb examples and frozen before TinyStories outcomes are read. No model calls or new chart choices are introduced.

## Locked hypotheses

- **F0_leave_one_pair_out_order_is_stable: supported.**
- **F1_fineweb_selected_order_transfers_to_tinystories: supported.**
- **F2_leave_one_pair_out_order_stays_near_the_order_oracle: supported.**
- **F3_optimal_description_order_differs_by_layer: supported.**
- **F4_cubic_is_an_adequate_universal_default: supported.**
- **F5_midpoint_expansion_is_more_stable_than_endpoint_integration: supported.**
- **F6_fifth_order_improves_support_even_when_behavior_oscillates: supported.**
- **F7_transfer_choices_are_frozen_before_test_outcomes: supported.**

| Layer | FineWeb global order/fidelity | LOO modal order/fraction | LOO routed/oracle/gap (95% CI) | TinyStories frozen/explicit/difference |
|---:|---:|---:|---:|---:|
| 0 | 3 / 0.817 | 3 / 1.000 | 0.817 / 0.852 / 0.015 (0.007, 0.144) | 0.778 / 0.768 / +0.010 |
| 1 | 5 / 0.736 | 5 / 0.941 | 0.724 / 0.807 / 0.044 (0.000, 0.065) | 0.755 / 0.742 / +0.013 |

A stable frozen order supports a compact interpreter program. Layer-dependent order remains a real complexity cost: cubic is the best cheap Layer-0 description, while Layer 1 benefits from fifth order. Endpoint-integral deterioration is a convergence-radius result, not evidence that the differentiated tensors are numerically wrong.
