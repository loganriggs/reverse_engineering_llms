## H750 — The verified operation coordinates have distributed upstream contributions

This experiment exactly expands the residual stream entering L13 attention into embedding/skip injection plus the attention and MLP outputs of layers0–12. It includes every learned residual multiplier. Each contribution is projected through L13H0's second query/key weights and the observed native RMS normalization factors, then into the frozen H742 rank-four coordinates. Nothing is fitted. Binding-to-copy coordinate differences are decomposed at the final query and source key on all H747 contexts.

An alignment share is the sum over pairs of dot(component difference,total coordinate difference), divided by the summed squared total difference. Shares sum to one but can be negative or exceed one. They are an algebraic attribution of the observed contrast, not causal effect fractions or explained-variance percentages. Each context uses its own observed normalization denominators, so denominator changes are distributed across these terms.

| Direct contribution on competent narratives | Query coordinates | Key coordinates |
|---|---:|---:|
| MLP outputs, L0–12 combined | 57.94% | 83.56% |
| Attention outputs, L0–12 combined | 42.07% | 16.72% |
| Embedding and skip injections | -0.001% | -0.274% |

The largest individual query contribution is L11 attention16.71%, followed by L9 attention9.95%, L9 MLP9.10%, L12 MLP8.16%, L10 MLP7.68%, L7 MLP7.36%, L11 MLP7.02%, L8 MLP6.99%. The largest key contributor is L6 MLP13.85%, followed by L9 MLP11.09%, L11 MLP9.79%, L12 MLP9.51%, L7 MLP9.32%, L8 MLP8.99%, L5 MLP8.31%, L10 MLP7.70%.

No single direct output dominates the component. MLP terms accumulate across middle and later layers, while query coordinates also receive appreciable late attention contributions. This suggests distributed construction rather than a single module writing the whole operation vector. However, later outputs already contain their upstream dependencies: the decomposition does not identify the earliest origin or prove that these outputs are necessary. Several outputs may reflect the same upstream cause.

Diagnostic lists are retained separately. Their largest query contributors are L7 MLP17.89%, L9 attention10.22%, L9 MLP9.74%, L11 attention9.11%; key contributors include L9 MLP19.96%, L6 MLP16.06%, L8 MLP11.13%. These cells still fail native binding competence and cannot establish successful generalization.

Verification:240 native model forwards; all hashes unchanged; native logits replay exactly; residual reconstruction maximum relative error1.55e-7; projected-coordinate reconstruction maximum absolute error1.83e-6; aggregate share-sum error2.13e-8. Both normalization levels and learned residual multipliers are included. No original reserved-test evaluations; natural document count26,112 unchanged.

Next causal test: swap earlier attention/MLP outputs between matched operation contexts and measure the resulting compact-coordinate and behavioral changes. Repeat while clamping L13H0's selected query/key states to native. The unclamped-minus-clamped effect tests mediation through these nodes, while exposing effects through other paths. Use a systematic layer/component sweep rather than treating the largest algebraic term as the causal winner. Compare attributions with causal results before training a featurizer on a presumed upstream site.
