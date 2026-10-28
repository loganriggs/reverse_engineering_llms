# Same variable, different finite functions

The source variable `function_word` is held fixed. In Layer 0, all edges carrying that variable are removed separately from Heads 3, 8, 7, and 6. The complete nonlinear suffix is recomputed. Near (`distance < 4`) and far invocations are also removed separately to compare variation within one native function against variation between functions.

## Locked hypotheses

- **D1_each_function_has_finite_downstream_effect: supported.**
- **D2_same_variable_has_distinct_functions: supported.**
- **D3_function_is_more_stable_than_invocation: not supported.**
- **D4_weight_distinction_survives_nonlinear_suffix: supported.**

| Head function | All-route mean KL | Near/far effect cosine |
|---|---:|---:|
| L0H3 | 1.330e-02 | 0.003 |
| L0H8 | 2.313e-03 | 0.079 |
| L0H7 | 2.953e-04 | 0.267 |
| L0H6 | 2.164e-04 | 0.075 |

Median absolute cross-head all-route cosine: `0.113`.

Cross-head all-route pairs:

- L0H3 vs L0H8: `+0.306`
- L0H3 vs L0H7: `+0.127`
- L0H3 vs L0H6: `+0.102`
- L0H8 vs L0H7: `+0.123`
- L0H8 vs L0H6: `+0.064`
- L0H7 vs L0H6: `+0.099`

## Guardrails

Removing all matching edges is a broad finite intervention, not a minimal semantic edit. Different downstream effects can arise from different routing weights, source values, or OV transforms; the weight atlas isolates OV distinction while this experiment checks that distinction survives the suffix. Near/far effects need not align even for one head because they carry different token states, so failure of the within-head comparison would not erase the cross-head distinction.
