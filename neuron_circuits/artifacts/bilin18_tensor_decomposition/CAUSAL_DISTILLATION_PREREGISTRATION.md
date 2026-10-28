# Causal CP distillation preregistration

This iteration follows the frozen decomposition scorecard. It tests whether a
CP-2048 layer can be made into a faithful causal surrogate by optimizing the
downstream behavior that the scorecard measures.

## Frozen comparison

All students begin from the same activation-refined CP-2048 checkpoint. The
base transformer is frozen. The training rows end at FineWeb row 15,999,
validation uses rows 17,000–17,007, and the previous scorecard begins at row
18,064.

## Variants

1. **KL:** match the complete downstream next-token distribution on ordinary
   language inputs.
2. **Causal:** match ordinary and intervened distributions and their finite
   logit change in the teacher's softmax-Fisher metric.
3. **Causal + tensor:** add a small isotropic bilinear-tensor loss with weight
   0.01 to the causal objective.

Gaussian RMS scales are sampled from 0.02, 0.05, 0.10, and 0.20. Natural
activation-mixture scales are sampled from 0.05, 0.10, 0.25, and 0.50.

## Hypotheses

- **H1:** KL distillation improves FineWeb KL without an OOD penalty greater
  than 0.005 on either corpus.
- **H2:** causal training improves Fisher effect fidelity for both primary
  intervention families relative to activation CP-2048.
- **H3:** the 0.01 tensor anchor improves isotropic tensor fidelity while
  costing at most 0.005 KL and 0.05 causal fidelity.
- **H4:** causal improvement transfers across intervention scales and OOD
  corpora.
- **H5:** at least one model reaches the existing 0.90 fidelity and 0.95 cosine
  causal gates on Gaussian 0.10 and natural-mix 0.25.

The intervention losses are scientific probes as well as optimization targets.
A model trained on them is not evidence that the same coordinates are a
canonical decomposition. Factor drift and semantic transfer remain separate
questions.
