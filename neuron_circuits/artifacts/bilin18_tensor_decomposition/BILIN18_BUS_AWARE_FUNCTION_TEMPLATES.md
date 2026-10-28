# Bus-aware reusable function templates

Each Layer-0--5 head is represented by its exact value-side map from two arguments—the current normalized residual and the persistent Layer-0 value bus—to the residual write. This corrects the current-value-only OV atlas. Full operator PCA, leading output-verb recurrence, and natural-response clustering test three different meanings of a reusable function template.

## Locked hypotheses

- **T1_full_bus_aware_operators_have_small_template_bank: not supported.**
- **T2_output_verbs_recur_beyond_random_geometry: supported.**
- **T3_natural_function_geometry_is_split_stable: supported.**
- **T4_natural_function_clusters_are_stable: not supported.**
- **T5_value_bus_materially_changes_function_geometry: not supported.**

## Summary

- Full weight-operator rank90: `22` of 54; top-16 energy `0.779`; participation ratio `21.4`.
- Median bus-input Frobenius fraction: `0.007`.
- Current-only versus bus-aware similarity Spearman: `1.000`.
- Leading output-verb median nearest cosine: `0.222` versus random p95 `0.077`.
- Natural-response geometry A/B correlation: `0.996`; cluster NMI `0.558`.
- Natural-response rank90 A/B: `45/45`.

Strongest leading-output neighbors:

- L4H2 ↔ L5H7: `0.705`
- L5H7 ↔ L4H2: `0.705`
- L1H1 ↔ L5H7: `0.641`
- L2H4 ↔ L5H0: `0.448`
- L5H0 ↔ L2H4: `0.448`
- L1H5 ↔ L4H1: `0.383`
- L4H1 ↔ L1H5: `0.383`
- L2H6 ↔ L3H6: `0.374`
- L3H6 ↔ L2H6: `0.374`
- L4H7 ↔ L5H6: `0.349`
- L5H6 ↔ L4H7: `0.349`
- L3H7 ↔ L4H3: `0.339`

Discovery-A natural-response clusters:

- C0: L0H2, L0H6, L1H0, L1H4, L2H0, L2H3, L5H8
- C1: L0H8, L3H1, L3H3, L4H2, L4H8, L5H0, L5H7
- C2: L0H0, L0H3, L1H3, L1H5, L1H7, L2H2, L2H6, L3H7, L4H7
- C3: L1H6, L2H1, L3H2
- C4: L0H5, L1H1, L3H0, L3H6, L4H0, L4H1, L4H3, L4H4, L5H6
- C5: L2H8, L3H4, L4H5, L5H1, L5H3, L5H5
- C6: L0H1, L0H4, L0H7, L1H2, L1H8, L2H4, L2H7, L5H2, L5H4
- C7: L2H5, L3H5, L3H8, L4H6

## Guardrails

The full bus-aware map is exact only for the value/write branch; QK routing, RMSNorm, and downstream computation remain outside it. Linear template rank, leading output directions, and clusters are distinct ontologies and need not agree. Natural signatures use a frozen random output sketch and aligned token positions but different texts across splits. Stable clustering would nominate function families, not prove semantic readability or causal interchangeability. Any surviving family requires a fresh finite replacement test.
