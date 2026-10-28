# Target-specificity and packing red-team of the midpoint router

Every method is reevaluated serially in the same three-member batch `(base, full, candidate)`. This removes method-count-dependent matrix-kernel variation. Correct and cyclic-wrong targets receive identical midpoint computation budgets; both select channels that are patched with the actual correct-target finite deltas.

## Locked hypotheses

- **B0_batch_packing_is_stable: supported.**
- **B1_midpoint_beats_base_activity_clustered: supported.**
- **B2_midpoint_is_target_specific: supported.**
- **B3_midpoint_support_is_target_specific: supported.**
- **B4_midpoint_nearly_matches_realized_oracle: supported.**
- **B5_base_tangent_specificity_is_positive_but_small: not supported.**

| Layer | Corpus | Serial activity/base/mid/wrong-base/wrong-mid/oracle | Mid/wrong support | Mid-activity pair median (95% CI) | Base target-specificity median (95% CI) | Max packing p95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.436/0.331/0.645/0.285/0.272/0.697 | 0.848/0.118 | +0.219 (+0.016, +0.414) | +0.022 (-0.084, +0.195) | 0.003 |
| 0 | tinystories | 0.339/0.293/0.692/0.295/0.160/0.695 | 0.845/0.138 | +0.345 (+0.099, +0.492) | +0.058 (-0.218, +0.239) | 0.009 |
| 1 | fineweb | 0.309/0.321/0.704/0.316/0.295/0.700 | 0.704/0.125 | +0.380 (+0.112, +0.534) | +0.122 (-0.085, +0.164) | 0.002 |
| 1 | tinystories | 0.475/0.333/0.753/0.198/0.254/0.752 | 0.718/0.127 | +0.305 (+0.225, +0.430) | +0.114 (-0.021, +0.210) | 0.002 |

The midpoint result is a target-conditioned circuit compiler, not a context-only feature detector: replacing the requested target while holding the base text fixed destroys both causal fidelity and native-support overlap.
