# Destination-aware operator compatibility atlas

Every non-target early operator is converted into an exact norm-and-angle-matched replacement for L3H8 under its native router. Finite KL is measured at scale 0.05 on two held-out halves. Candidate scores are frozen from the earlier bus-aware weight and natural-response panels; no evaluation KL enters them.

## Locked hypotheses

- **A1_compatibility_ranking_is_split_stable: supported.**
- **A2_locked_joint_geometry_predicts_compatibility: not supported.**
- **A3_natural_response_similarity_predicts_compatibility: not supported.**
- **A4_output_direction_similarity_predicts_compatibility: not supported.**
- **A5_geometry_selected_partner_is_causally_best: supported.**

Finite compatibility rank correlation across halves: `0.863`.

| Independent feature | Spearman with −KL A/B |
|---|---:|
| joint_geometry_score | -0.112 / -0.012 |
| replicated_natural_abs_cosine | -0.035 / -0.013 |
| output_verb_abs_cosine | 0.325 / 0.296 |
| full_operator_abs_cosine | -0.130 / -0.025 |

Best finite replacements:

- L2H5: KL `5.919e-06/2.703e-06`, joint geometry `0.0783`, natural `0.373`, output `0.280`
- L5H1: KL `6.826e-06/3.415e-06`, joint geometry `0.0000`, natural `0.002`, output `0.053`
- L1H6: KL `6.909e-06/3.333e-06`, joint geometry `0.0000`, natural `0.001`, output `0.047`
- L5H4: KL `6.925e-06/3.339e-06`, joint geometry `0.0000`, natural `0.022`, output `0.030`
- L1H0: KL `6.897e-06/3.495e-06`, joint geometry `0.0000`, natural `0.011`, output `0.052`
- L5H8: KL `6.967e-06/3.543e-06`, joint geometry `0.0000`, natural `0.004`, output `0.013`
- L5H6: KL `7.086e-06/3.542e-06`, joint geometry `0.0002`, natural `0.014`, output `0.036`
- L1H3: KL `7.170e-06/3.590e-06`, joint geometry `0.0000`, natural `0.010`, output `0.007`
- L5H3: KL `7.072e-06/3.722e-06`, joint geometry `0.0000`, natural `0.005`, output `0.055`
- L4H1: KL `7.254e-06/3.542e-06`, joint geometry `0.0000`, natural `0.007`, output `0.021`
- L5H0: KL `7.510e-06/3.287e-06`, joint geometry `0.0000`, natural `0.006`, output `0.013`
- L4H6: KL `7.208e-06/3.837e-06`, joint geometry `0.0002`, natural `0.017`, output `0.114`

## Guardrails

The candidate score chose L2H5 before this finite panel, but its formula was motivated by prior results and is not a universal compatibility law. All replacements retain the target router and destination. A correlation failure would mean the successful pair is an outlier, not that its finite advantage is false; success would nominate a prospective metric for other targets that still requires fresh-target validation.
