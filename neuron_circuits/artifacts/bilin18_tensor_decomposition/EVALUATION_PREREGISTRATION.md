# Bilin18 layer-1 decomposition evaluation preregistration

This document fixes the evaluation questions and directional hypotheses before
running the expanded decomposition benchmark. It supplements the exploratory
results in `REPORT.md`.

## Scientific target

The target is not the decomposition with the highest reconstruction score in
isolation. It is the shortest description that faithfully predicts both normal
behavior and finite interventions. Global weight fidelity, language-manifold
fidelity, causal fidelity, sparsity, stability, and semantic transfer are kept
as separate measurements.

## Primary models

1. Direct weight-fitted CP-2048.
2. Activation-refined CP-2048.
3. Mixed tensor-plus-behavior CP-2048 with Top-K 128 routing.

Activation-refined CP-512 and activation-refined LL1 R256/L8 are negative or
lower-capacity controls. Existing checkpoints are frozen for this evaluation.

## Metrics

- **Behavior:** cross-entropy change, teacher-to-student KL, and top-1 agreement,
  with sequence-bootstrap confidence intervals.
- **OOD behavior:** the same metrics on WikiText-103 and TinyStories.
- **Causal fidelity:** agreement between teacher and replacement changes in
  full-model log probabilities after the same intervention is applied to the
  layer-1 MLP input. Interventions are Gaussian RMS 0.10 perturbations and a
  25% move toward a position-shuffled natural activation.
- **Parsimony:** stored parameter count and bytes, approximate multiply count,
  active factors per token, corpus-wide effective support, and routing entropy.
- **Stability:** symmetry-aware CP-term similarity and activation-space CKA
  between direct, activation-refined, and routed CP-2048 checkpoints.
- **Semantic proxy:** token-form categories are selected using FineWeb only and
  evaluated on held-out FineWeb, WikiText-103, and TinyStories. This restricted
  vocabulary cannot establish complete interpretability.

## Directional hypotheses

- **H1 — manifold gap:** activation refinement improves natural behavior while
  reducing isotropic tensor fidelity.
- **H2 — causal gap:** its behavioral advantage over direct CP shrinks on OOD
  data or MLP-input interventions.
- **H3 — sparse conditional program:** routed CP uses at most 6.25% of its
  factors per token but more than half of them across a corpus.
- **H4 — collective stability:** activation-space similarity is higher than
  individual-term stability across objectives.
- **H5 — semantic transfer:** FineWeb-selected token-form explanations average
  oriented AUROC above 0.60 on both OOD corpora.
- **H6 — no current full pass:** no frozen candidate passes every behavioral,
  OOD, causal, and parsimony gate.

## Proposed engineering gates

- FineWeb mean KL no more than 0.02.
- Mean KL no more than 0.05 on each OOD corpus.
- Top-1 agreement at least 95% on every corpus.
- Causal effect fidelity at least 0.90 and effect cosine at least 0.95 for both
  intervention families.

These are deliberately demanding proposed gates, not universal standards. A
model that misses them can still identify an informative Pareto frontier.
