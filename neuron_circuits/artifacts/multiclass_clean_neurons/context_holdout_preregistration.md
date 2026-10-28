# Context-holdout confirmation preregistration

Sealed on 2026-10-17 after reading the Round 4 result and before running any context-holdout experiment.

## Motivation

Round 4 used fresh noise samples but the discovery set covered every class and operation. A monosemantic interpretation should generalize across contexts: an operation/“verb” direction should retain its effect on unseen classes, and a class/“noun” direction should retain its effect under unseen operations.

## Frozen change

All Round 4 settings and endpoints remain fixed, except for disjoint context partitions:

- Operation directions: fit on classes 0–49, validate on 50–74, confirm on 75–99.
- Class directions: fit on operations 0–3, validate on 4–5, confirm on 6–7.
- The confirmation examples are also independently generated noisy samples.
- Experiment script SHA-256: `b234142ae8a280788f1e0a4aa6e9672621b23a1130001aeca056a16e52a0ab8d`.
- Seeds 0–9 and 10 random rotated bases per seed.

Exact command is the Round 4 template plus `--context-holdout`, writing to `context_holdout100/seed_SEED`.

## Locked decisions

The same sparse-curve endpoints and thresholds are used:

- strong clean axes: axis-privilege 95% CI lower bound above `+0.10`;
- practically negligible axes: axis-privilege 95% CI upper bound below `+0.05`;
- distributed reusable subspace: PCA-minus-raw 95% CI lower bound above `+0.10`.

All families, layers, failed runs, and ranks will be disclosed.

