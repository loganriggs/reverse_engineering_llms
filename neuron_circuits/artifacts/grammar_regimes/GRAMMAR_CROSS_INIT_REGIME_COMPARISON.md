# Cross-initialization recurrence: pure rules versus mixed memory

This is a post-hoc comparison of separately locked assays, not a new preregistered hypothesis test. Only candidates passing 0.97 trained-template recall and 0.95 unseen-template transfer are included (two candidate initializations per regime, each compared with the same reference).

| Quantity | Pure rule | 50% arbitrary |
|---|---:|---:|
| Top-64 effect-subspace overlap | 0.119 | 0.038 |
| Random-subspace expectation | 0.062 | 0.031 |
| Chance-adjusted top-64 overlap | 0.060 | 0.007 |
| Complete natural-effect cosine | 0.159 | 0.005 |
| Matched native-channel held-out cosine | 0.033 | -0.001 |
| Mapped per-site top-32 overlap | 0.188 | 0.166 |
| Late gate-map R2 (Layer 3) | 0.612 | -2.604 |
| Late full-Jacobian cosine (boundary 4) | 0.242 | 0.084 |

## Interpretation

Pure reuse produces more cross-run recurrence, especially in late distributed gate states, but not canonical one-neuron correspondences. Mixed memory largely destroys even that transferable gate coordinate map. Yet neither regime admits a global linear conjugacy of its complete local answer Jacobian. This is compatible with the earlier positive result that selected pure-rule semantic edit vectors transfer causally after alignment: the recurring object is a task-conditioned finite program, not every possible local perturbation.

## Red-team boundary

The pure and mixed discovery panels have different sizes because mixed panels balance selected and unselected facts. The subspace comparison therefore reports the analytic random-subspace expectation and a chance-adjusted value. There are only two gate-passing candidate initializations per regime, and checkpoint training durations differ. Treat the magnitude as exploratory; the sign agrees with the independently established within-model semantic-transfer gap. Negative full-Jacobian alignment does not show that the models compute different answers or that no nonlinear/contextual alignment exists.
