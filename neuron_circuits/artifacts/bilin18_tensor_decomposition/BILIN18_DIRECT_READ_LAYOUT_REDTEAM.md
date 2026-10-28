# Direct-read operator transfer across altered layouts

Neutral material is inserted immediately before the original prediction-position token, increasing the relative distance from the edited operand while preserving the original answer. Mechanistic results are reported only for cells with at least two pairs whose base and source endpoints both retain exact full-vocabulary top-1 behavior.

## Locked hypotheses

- **L0_layouts_retain_exact_behavior: supported.** Observed: `{'families': ['agreement_attractor', 'ioi', 'repeated_sequence']}`
- **L1_native_edited_edges_survive_layout_change: supported.** Observed: `{'families': ['agreement_attractor', 'ioi', 'repeated_sequence']}`
- **L2_original_frozen_routes_transfer_across_layouts: supported.** Observed: `{'families': ['agreement_attractor', 'ioi', 'repeated_sequence']}`
- **L3_two_token_distance_shift_transfers: supported.** Observed: `{'families': ['agreement_attractor', 'ioi', 'repeated_sequence']}`
- **L4_route_coefficients_are_layout_stable: supported.** Observed: `{'mean_cosine': 0.9660002635363029, 'cells': 6}`

| Family | Added tokens | Exact pairs | Native edited fidelity | Original frozen-route fidelity | Route cosine |
|---|---:|---:|---:|---:|---:|
| agreement_attractor | 1 | 2/6 | 0.868 | 0.714 | 0.955 |
| agreement_attractor | 2 | 2/6 | 0.835 | 0.749 | 0.965 |
| ioi | 1 | 5/6 | 0.997 | 0.998 | 0.999 |
| ioi | 2 | 5/6 | 0.998 | 0.998 | 1.000 |
| capital_recall | 1 | 0/3 | — | — | — |
| capital_recall | 2 | 0/3 | — | — | — |
| repeated_sequence | 1 | 4/4 | 1.000 | 0.856 | 0.949 |
| repeated_sequence | 2 | 2/4 | 1.000 | 0.930 | 0.928 |

## Guardrail

These variants alter relative distance but preserve the same broad syntactic template. Success rules out a single absolute-position lookup; it does not yet establish transfer to arbitrary paraphrases. Cells losing exact behavior are capability failures and are not used as negative mechanistic evidence.
