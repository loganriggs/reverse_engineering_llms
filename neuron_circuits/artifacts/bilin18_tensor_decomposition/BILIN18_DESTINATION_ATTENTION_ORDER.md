# Attention-composition order in destination compatibility

After the preregistered single-attention program failed, this follow-up groups the exact analytic tensor expansion by the number of downstream attention Jacobians in a path. Order 0 is the attention-free residual/MLP route; order 1 sums every path with exactly one downstream attention use; higher orders include increasingly long attention-mediated chains. No finite outcomes are used in this decomposition.

## Locked hypotheses

- **O1_second_order_materially_improves_all_targets: not supported.**
- **O2_order_six_is_sufficient: not supported.**
- **O3_earlier_destinations_require_higher_order: not supported.**
- **O4_multi_attention_composition_is_necessary: supported.**
- **O5_order_energy_profile_is_split_stable: not supported.**
- **attention_jvp_forward_invariant: supported.**

## Cumulative order frontier

| Target / half | First ≥.90 | order 0 | order 1 | order 2 | order 3 | order 4 | order 5 | order 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0H0 / a | None | 0.036 | -2.553 | -16.671 | -92.535 | -255.156 | -391.084 | -344.122 |
| L0H0 / b | None | 0.158 | -0.350 | -3.932 | -12.083 | -24.433 | -29.189 | -20.122 |
| L0H0 energy split stability | 0.964 | — | — | — | — | — | — | — |
| L2H2 / a | None | 0.113 | 0.121 | -0.236 | -2.042 | -4.399 | -4.748 | -2.359 |
| L2H2 / b | None | -0.040 | 0.285 | 0.208 | -0.054 | 0.057 | 0.382 | 0.702 |
| L2H2 energy split stability | 0.607 | — | — | — | — | — | — | — |
| L5H3 / a | 5 | 0.451 | 0.816 | 0.802 | 0.771 | 0.849 | 0.950 | 0.992 |
| L5H3 / b | 5 | 0.545 | 0.787 | 0.766 | 0.780 | 0.895 | 0.977 | 0.998 |
| L5H3 energy split stability | 0.929 | — | — | — | — | — | — | — |

## Guardrails

Orders are algebraic path groups in the local Jacobian, not claims that the model executes a discrete sequence of symbolic calls. Cumulative fidelity can be non-monotonic because different path orders cancel. An early layer has more possible downstream attention operations than a late layer, so depth and learned organization remain partially confounded. The split test addresses sampling stability, not that structural confound.
