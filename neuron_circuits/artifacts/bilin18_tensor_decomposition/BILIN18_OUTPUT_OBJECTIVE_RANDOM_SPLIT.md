# Random-split red team of the sparse output basis

Eight seeded random half-vocabulary splits replace the original even/odd split. Varimax is fitted without labels on each training half. Held-out sparsity and semantic label search are compared with sixteen matched random rotations per split.

Median/minimum pairwise recovered-axis overlap: **0.994/0.989**.
Median held-out kurtosis and advantage over random p95: **2.875/+2.134**.
Median held-out semantic AUROC and advantage over random p95: **0.780/-0.053**.

The random splits test whether tokenizer-ID parity manufactured the stable sparse coordinates. They remain a post-hoc robustness check, not a new preregistered semantic claim.
