# Capacity sweep: preregistered questions and metrics

This file was written before inspecting any new capacity-sweep results. The task is held fixed at four transformer layers, 50% selected arbitrary assignments, 128 training facts per relation/answer stratum, and the same data manifest within each seed. Width is varied over 64, 96, 192, and 384. SwiGLU is primary; the symmetric bilinear MLP is a replication.

## Questions

1. Does added width improve recall of changed arbitrary facts, generalization of the remaining structured rule, or both?
2. Does added width create a privileged neuron basis, or only a sparse subspace that looks equally sparse after random rotation?
3. Are small causal circuits reused by examples with the same semantic rule, rather than merely being sparse per-example lookup circuits?

Only facts the model predicts correctly are used for circuit-size comparisons. Results are reported separately for learned changed training facts, learned unselected training facts, and correctly generalized held-out structured facts.

## Causal circuit counts at 90%

A neuron means one post-product MLP channel in one layer, applied at every token position. Thus a four-layer model has `4 * 2 * width` candidate neurons.

For each example, let `m_full` be the correct-answer margin of the intact model and `m_zero` the margin when every MLP channel is zeroed. Rank channels using the absolute first-order contribution `|sum_position activation * gradient_of_margin|`. All reported subset effects are then measured by exact forward passes with downstream computation and final normalization recomputed.

- **S90 (sufficient circuit size):** the smallest tested top-ranked set whose exact margin recovers at least 90% of the MLP-added margin, `(m_keep - m_zero) / (m_full - m_zero) >= 0.9`.
- **N90 (necessary removal size):** the smallest tested top-ranked set whose removal erases at least 90% of the MLP-added margin, `(m_full - m_remove) / (m_full - m_zero) >= 0.9`.

Counts are undefined when the intact prediction is wrong or when `m_full - m_zero <= 0`; exclusions and denominators will be reported. Because nonlinear subset effects need not be monotonic, these are ranked-circuit counts, not proofs of globally minimal circuits. The implementation evaluates a fixed logarithmic grid and refines the first observed crossing.

## Why S90 or N90 alone is not monosemanticity

A fact-specific lookup can have a one-neuron S90 while carrying no reusable concept. Conversely, a semantic computation may be distributed across several neurons. We therefore call the full measurement a **Causal Monosemanticity Profile at 90% (CMP-90)** rather than reducing it immediately to one score:

1. median S90 count and fraction of all channels;
2. median N90 count and fraction;
3. **axis advantage**, `log2((rotated S90 + 1) / (native S90 + 1))`, under fixed random orthogonal rotations of each layer's MLP channel basis;
4. **semantic reuse**, the fraction of same-rule recipients for which a donor's S90 mask recovers at least 90% of the recipient's MLP-added margin;
5. matched-answer exception reuse as a non-semantic control.

A strong neuron-aligned semantic regime should have small native S90, larger rotated S90, and high same-rule reuse. A sparse memorization regime may have small native S90 but low semantic reuse. A distributed constraint regime may have larger S90/N90 and weak axis advantage while still admitting accurate parameter-level explanations.

## Directional hypotheses

- If width separates reusable features from arbitrary exceptions, structured generalization, native axis advantage, and same-rule reuse should rise with width at matched learned-fact status.
- If width merely adds redundant coordinates, S90 as a fraction may fall while absolute S90, rotation sensitivity, and semantic reuse do not improve.
- If exception storage becomes localized lookup, exception S90 may shrink but matched-answer reuse should remain low.
- If the native neuron basis is not privileged, random rotations should preserve S90 up to sampling noise; sparse native counts then are not evidence for neuron-aligned concepts.

No single observation is designated as decisive. Behavior, exact circuit effects, rotation controls, and reuse must agree before describing a regime as neuron-aligned and monosemantic.

## Logged red-team refinement

The first implementation paired structured donor and recipient examples by relation alone. Before interpreting that output, this was identified as a readout confound: the two examples could implement the same rule but require different answer values. The reported reuse test therefore matches structured pairs on both relation and resulting structured answer. Its exception control matches the resulting answer while requiring a different relation. Recipients are drawn from the full naturally correct pool rather than only the small circuit-discovery panel. The original files were overwritten by a full rerun; this note preserves the timing and reason for the change.

A second check found that 87% of the initial native sufficiency curves had a decrease larger than 0.05 somewhere among the evaluated prefixes. This falsifies the local-monotonicity assumption behind the proposed binary refinement. The final audit therefore evaluates a fixed grid: every count from 0 through 64, powers of two, and 65 evenly spaced counts through the full channel set. S90/N90 are the smallest *tested* prefixes crossing 90%, hence resolution-limited upper-bound estimates rather than exact minima. Non-monotonicity is itself reported as evidence of channel cooperation and cancellation.
