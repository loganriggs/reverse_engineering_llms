# Ground-truth computation-assay benchmark

## Purpose

Before a missing signature is used to disfavor a real-model mechanism, verify that the complete analysis can recover that signature when it is known to exist. This benchmark tests the metrics used by the hypothesis ledger, not whether synthetic mechanisms resemble all details of language models.

## Shared substrate

Use five seeds, 128 internal channels, a 64-dimensional state, a 32-dimensional downstream-effect vector, eight operation labels, and disjoint discovery/validation/test sets of 4,096 examples each. Generate each family at several signal-to-dense-background strengths.

All families expose the same observables:

- native channel activations and writes;
- context state and operation label;
- exact finite downstream effect;
- local Jacobian with respect to state;
- parameter-margin gradients for the constraint family.

Fit every explanation on discovery, choose hyperparameters on validation, and report only once on test.

## Planted families

### A. Native-axis modules

Each operation uses four disjoint native channels with context-dependent amplitudes and fixed downstream writes. A matched random orthogonal rotation plus inverse decoder preserves the input-output function and supplies the coordinate-control model.

Expected recovery: native S90 is at most four, native axis advantage exceeds 1 bit, and frozen same-operation transfer fidelity exceeds 0.95.

### B. Rotated reusable modules

Apply a fixed dense orthogonal mixing to the channel representation of family A and exactly compensate in the decoder. The function and semantic operations are unchanged, but individual coordinates are not privileged.

Expected recovery: a learned operation subspace reaches test effect fidelity above 0.95 at the planted rank; native coordinates do not reliably beat random rotations.

### C. Sparse payload with dense address

Construct a bank of 64 fixed downstream payload modes. Four modes are selected per example by continuous scores computed from all 32 address coordinates. The exact payload is sparse after the routing decision.

Expected recovery: oracle top-four fidelity exceeds 0.95; every fixed four-mode set is below 0.60; router fidelity rises with available address dimension and exceeds 0.90 only when supplied enough of the planted address.

### D. Local atlas

Use a four-dimensional payload whose orientation rotates smoothly with state through several noncommuting generators. Each local Jacobian and short path is rank four, but the union of charts spans most of the channel space.

Expected recovery: local Jacobian/path reconstruction exceeds 0.95; a fixed rank-four global subspace remains below 0.60 on held-out contexts. Increasing fixed rank should recover the atlas gradually.

### E. Dense computation null

Generate full-rank state-dependent effects with no planted sparse or low-rank organization beyond finite-sample structure.

Expected recovery: no rank-four native, rotated, PCA, routed, or local-chart explanation exceeds 0.60 held-out effect fidelity. Full-rank identity reconstruction must remain exact.

### F. Constraint geometries

Create two parameter-margin systems with identical gradient norms and self-effects:

- a taut correlated graph with signed low-rank-plus-sparse gradients;
- a diffuse nearly orthogonal graph with weak broad edges.

Apply exact finite source-gradient edits to linear margins and a separately generated mildly nonlinear version.

Expected recovery: the signed Gram kernel reaches Pearson and origin-anchored R-squared above 0.98 for the linear system and above 0.90 at the smallest nonlinear scale; a source-permutation control remains below 0.10. Coupling and normalized collateral rank taut above diffuse, while diffuse effective support ranks above taut.

## Required metrics

1. exact S90/N90 and random-rotation axis advantage;
2. frozen learned-subspace transfer;
3. oracle sparse support and fixed-subset control;
4. router support recall and downstream-effect fidelity;
5. local versus global chart reconstruction;
6. signed-kernel finite-edit prediction;
7. effect-space stability across independently mixed coordinates.

## Injection and power curve

Mix every planted effect with a matched dense background at signal fractions 1.0, 0.5, 0.25, 0.125, and 0.0625 while preserving total effect norm. Report the minimum fraction at which each locked signature is detected in at least four of five seeds.

A real-model negative can only exclude mechanisms stronger than this empirical detection boundary. We will not extrapolate below it.

Post-smoke clarification: these locked thresholds are high-fidelity sufficiency gates. Under mixtures they establish when the planted mechanism remains sufficient for the total effect, not the minimum strength at which its presence can be detected. Presence sensitivity requires a separate matched-null test and will not be inferred from these curves. Signal fractions 0.9 and 0.75 were added to resolve the observed sufficiency transition; the original fractions and thresholds are retained.

## Failure handling

If a family misses its expected recovery threshold:

1. classify this as assay failure rather than evidence against the family;
2. inspect identity reconstruction, split leakage, rotation compensation, and effect normalization;
3. compare an independent direct implementation of the metric;
4. revise the assay before applying the corresponding falsifier to the 500M model.
