# Does L9H8 transport dialect state across boundaries?

The weight-only leading mode suggested a US-geography versus British-spelling axis after L9H8 had independently been validated as a boundary reader. Prefixes containing only US or only UK spelling markers are matched on boundary token identity and log-distance. The first 10,000 corpus rows orient/fits directions; the second 10,000 rows are held out.

## Locked results

- **D0_model_exposes_dialect_state: supported.**
- **D1_boundary_leading_mode_encodes_dialect: supported.**
- **D2_boundary_full_write_encodes_dialect: supported.**
- **D3_boundary_edge_transmits_dialect_to_logits: supported.**
- **D4_address_controls_are_balanced: supported.**
- **D5_dialect_interpretation_is_supported: supported.**

| Head | Native style AUC | leading weight mode | fitted write direction | finite edge style contribution | address control |
|---|---:|---:|---:|---:|---:|
| boundary_L9H8 | 0.904 | 0.820 | 0.891 | 0.818 | 0.491 |
| same_layer_L9H7 | 0.904 | 0.533 | 0.571 | 0.606 | 0.491 |
| L8H1 | 0.904 | 0.496 | 0.490 | 0.530 | 0.491 |

## Guardrails

The spelling vocabulary was selected after inspecting the L9H8 singular mode, so only the second corpus half tests generalization; this is not an unbiased search over possible semantic axes. Prefix labels are weak supervision: topics, source domains, and geography correlate with spelling. A fitted write probe shows encoded information but not a unique native coordinate. The finite US-minus-UK logit contrast is causal for that declared output axis, not proof that dialect is the head's only function.
