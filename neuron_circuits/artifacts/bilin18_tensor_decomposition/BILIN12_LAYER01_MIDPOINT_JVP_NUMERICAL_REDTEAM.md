# Numerical red-team of the midpoint product JVP

A centered finite-difference sweep separates truncation error at large epsilon from float32 cancellation at small epsilon. The JVP implementation and examples are unchanged from the adaptive-chart experiment; only the numerical comparator step is varied.

## Locked hypotheses

- **N0_one_shared_epsilon_validates_the_jvp: supported.**
- **N1_the_failed_epsilon_is_roundoff_limited: supported.**
- **N2_direction_is_stable_across_the_sweep: not supported.**
- **N3_no_cell_requires_a_different_epsilon_regime: not supported.**

| Layer | Corpus | epsilon: min cosine / median error / max error | Best epsilon |
|---:|---|---|---:|
| 0 | fineweb | 0.1: 0.999833/0.0125/0.0327; 0.03: 0.999999/0.0012/0.0030; 0.01: 0.999992/0.0008/0.0040; 0.003: 0.999912/0.0028/0.0133; 0.001: 0.999134/0.0084/0.0416 | 0.03 |
| 0 | tinystories | 0.1: 0.999810/0.0128/0.0385; 0.03: 0.999998/0.0012/0.0035; 0.01: 0.999998/0.0009/0.0022; 0.003: 0.999976/0.0026/0.0070; 0.001: 0.999759/0.0075/0.0221 | 0.01 |
| 1 | fineweb | 0.1: 0.996269/0.0108/0.0979; 0.03: 0.999937/0.0013/0.0124; 0.01: 0.999994/0.0017/0.0033; 0.003: 0.999936/0.0051/0.0113; 0.001: 0.999622/0.0148/0.0276 | 0.01 |
| 1 | tinystories | 0.1: 0.992930/0.0096/0.2798; 0.03: 0.999931/0.0011/0.0270; 0.01: 0.999996/0.0013/0.0031; 0.003: 0.999908/0.0043/0.0136; 0.001: 0.999598/0.0133/0.0284 | 0.01 |

A failed small-epsilon amplitude gate does not invalidate the analytic JVP if the finite difference converges toward it at a larger shared epsilon while direction remains stable. The original preregistered gate remains recorded as failed.
