# Exact-distance synthetic control for midpoint compilation

For every semantic source→target edit, two random endpoints are constructed on the normalized embedding sphere at exactly the same chord distance from the source. They cannot encode the declared lexical relation. Semantic and synthetic edits use the same source contexts, rank, midpoint-JVP budget, finite-delta patch, and output metric.

## Locked hypotheses

- **Y0_semantic_edits_compile_better_than_synthetic: not supported.**
- **Y1_semantic_edits_have_more_predictable_support: not supported.**
- **Y2_synthetic_edits_also_compile: not supported.**
- **Y3_synthetic_direction_is_predictable: not supported.**
- **Y4_synthetic_distance_is_exact: supported.**
- **Y5_semantic_fidelity_advantage_survives_pair_bootstrap: not supported.**

| Layer | Corpus | Semantic midpoint/oracle/support/cosine | Synthetic midpoint/oracle/support/cosine | Semantic-synthetic pair median (95% CI) |
|---:|---|---:|---:|---:|
| 0 | fineweb | 0.679/0.717/0.872/0.996 | 0.655/0.664/0.831/0.992 | +0.037 (-0.112, +0.266) |
| 0 | tinystories | 0.711/0.702/0.858/0.995 | 0.644/0.691/0.794/0.989 | +0.026 (-0.160, +0.396) |
| 1 | fineweb | 0.697/0.728/0.701/0.976 | 0.653/0.736/0.609/0.940 | -0.019 (-0.195, +0.160) |
| 1 | tinystories | 0.679/0.681/0.721/0.980 | 0.533/0.672/0.588/0.937 | +0.066 (-0.082, +0.558) |

Maximum relative edit-distance mismatch: **1.21e-07**.

Success on synthetic endpoints would identify generic low-degree tensor geometry, not semantic abstraction. A semantic advantage in support but not fidelity would instead mean natural edits are more natively axis-aligned without being easier to reproduce behaviorally.

## Post-run pair-clustered support diagnostic

The locked hypotheses required a universal 0.05 median support advantage but did
not attach clustered uncertainty to that comparison. A post-run 20,000-sample
lexical-pair bootstrap gives semantic-minus-synthetic support advantages:

| Layer | Corpus | Pair median | 95% CI |
|---:|---|---:|---:|
| 0 | fineweb | +0.065 | (+0.024, +0.114) |
| 0 | tinystories | +0.057 | (+0.005, +0.103) |
| 1 | fineweb | +0.077 | (+0.051, +0.136) |
| 1 | tinystories | +0.125 | (+0.067, +0.152) |

Direction-cosine advantages are also positive in every cell, but causal-fidelity
intervals cross zero. Because the exact-distance controls are off the token
manifold, this supports a natural-manifold/native-support advantage rather than
isolating semantic relation structure specifically.
