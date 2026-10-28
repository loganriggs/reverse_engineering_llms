# Preregistration: best-of-many effect-collision null

The sitewise atlas chooses the best of 84 attention-layer triples after seeing
the finite target effect. Because the model's state-conditioned output effect
space is low-rank, this could be a multiple-comparison collision rather than a
localized attention route.

At every frozen site compare the best structured triple with:

- 84 uniformly random 16-mode Layer-8 programs;
- 84 random 16-mode programs sampled without replacement with probabilities
  proportional to each mode's full-Fisher diagonal divided by its input
  amplitude cost.

All candidate programs are evaluated through exact finite 5%-norm forward
interventions. Every finite batch contains its own full-reference intervention
to control GPU batch-shape roundoff. The maximum fidelity is computed only
after all 84 candidates are evaluated, matching the structured search budget.

The weighted null deliberately sees the full local metric. It is an adversarial
test of whether the *attention-mask structure* adds information beyond knowing
which individual modes are promising. It is not a deployable interpreter.

Structured advantage is locked at +0.05 median or +0.10 at the 10th percentile
over uniform, and +0.02 median or +0.05 at the 10th percentile over the stronger
weighted null, on both corpora. Collision equivalence uses tolerances 0.01 and
0.05 respectively. These are effect-space tests, not support-recovery tests.
