# Round 2 conclusion: replicated 10-class prototype task

This round was exploratory, not the final confirmation. Ten independently initialized models all reached 100% accuracy on fresh noisy prototype examples (mean CE `3.60e-5`).

The intervention was an operation counterfactual: preserve the content prototype and replace the operation code. Candidate directions were fitted only on discovery data, checked on validation data, and reported on fresh confirmation data.

## Result

Across 10 seeds, raw neurons had a small advantage over equally sparse coordinates in random orthogonal bases:

- Layer 1: `+0.008`, `+0.018`, and `+0.034` KL recovery at ranks 1, 4, and 16.
- Layer 2: `+0.003`, `+0.010`, and `+0.018` at the same ranks.

Learned PCA subspaces were much stronger than raw neurons:

- Layer 1 PCA-minus-raw: `+0.157`, `+0.384`, and `+0.508` at ranks 1, 4, and 16.
- Layer 2 PCA-minus-raw: `+0.152`, `+0.466`, and `+0.604`.

The numerical source is [replication_summary.json](round2_replication10/replication_summary.json).

## Reseal / red-team

This does not establish monosemantic neurons. The small raw-axis benefit is consistent with weak axis alignment, but most sparse causal information is much more compact in a learned distributed subspace.

Important limitations:

1. The operation identity has only seven non-identity values, so a compact operation-delta subspace is partly built into the task.
2. Only operation changes were studied; the result could be generic to the representation rather than specific to computation-like features.
3. The model has only 10 output classes.
4. Raw coordinates were ranked by activation-delta energy, not by an oracle combinatorial causal search. This is intentional for a scalable dataset method, but it is not the best possible sparse subset.
5. A counterfactual control circuit is not the same as a sufficient circuit: a tiny patch can redirect intact downstream machinery.

