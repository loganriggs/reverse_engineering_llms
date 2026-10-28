# Direct-read route quantization

The checkpoint weights, contextual values, three-head identity, and edited-source role are treated as shared machinery. Only the discovery-mean head-by-role route coefficients are quantized. This is a conditional description-length curve, not a standalone bit count for the whole circuit.

## Locked hypotheses

- **Q0_route_templates_are_small: supported.** Observed: `{'scalars': {'agreement_attractor': 3, 'ioi': 3, 'capital_recall': 3, 'repeated_sequence': 6}}`
- **Q1_four_bit_routes_retain_function: supported.** Observed: `{'fidelities': {'agreement_attractor': 0.9227690977512835, 'ioi': 0.9986259778054669, 'capital_recall': 0.998952641639373, 'repeated_sequence': 0.8946041340827007}}`
- **Q2_two_bit_routes_often_retain_function: supported.** Observed: `{'fidelities': {'agreement_attractor': 0.9218597425167072, 'ioi': 0.9901971676236536, 'capital_recall': 0.9996185392263643, 'repeated_sequence': 0.9151254981462413}}`
- **Q3_one_bit_sign_routes_are_not_assumed_sufficient: not supported.** Observed: `{'fidelities': {'agreement_attractor': 0.6860621759326981, 'ioi': 0.988129434594144, 'capital_recall': 0.8931647900114381, 'repeated_sequence': 0.8730003705408023}}`
- **Q4_quantized_routes_preserve_float_operator: supported.** Observed: `{'float_minus_four_bit': {'agreement_attractor': 6.882690787257673e-05, 'ioi': 2.0464711491419152e-05, 'capital_recall': -0.0003254374287869899, 'repeated_sequence': 0.0014508855460793058}}`

| Family | Route scalars | Float | 1 bit | 2 bit | 3 bit | 4 bit | 8 bit |
|---|---:|---:|---:|---:|---:|---:|---:|
| agreement_attractor | 3 | 0.923 | 0.686 | 0.922 | 0.923 | 0.923 | 0.923 |
| ioi | 3 | 0.999 | 0.988 | 0.990 | 0.997 | 0.999 | 0.999 |
| capital_recall | 3 | 0.999 | 0.893 | 1.000 | 0.999 | 0.999 | 0.999 |
| repeated_sequence | 6 | 0.896 | 0.873 | 0.915 | 0.891 | 0.895 | 0.896 |

## Interpretation boundary

This curve measures the extra metadata needed after the native tensor operator and contextual values are given. It should not be compared directly with a standalone compressed network. A short route template explains which typed operation is invoked; it does not encode the variable object carried in the value vector.
