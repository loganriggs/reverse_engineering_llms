# Preregistration: representational regimes in a compositional mini-language

Locked on 2026-10-18 before generating manifests, training models, or viewing
activations.

## Goal

Build an autoregressive language task that contains reusable grammar-like
computation and fixed arbitrary exceptions in the same model.  Determine which
explanatory object—native neuron, distributed semantic subspace, fact address,
or signed constraint network—best predicts held-out behavior and causal
interventions as depth, MLP type, load, and optimizer change.

This experiment is designed to bridge the existing entity-attribute pilot and
the user's FineWeb tensor transformers.  It is not intended to establish that
the synthetic grammar has the same statistics as natural language.

## Task

Entities are three-token names drawn from a shared eight-token alphabet.  No
entity gets a private embedding.  Three latent traits are deterministic
nonlinear modular functions of the name digits.  A sentence supplies two
entity names and one of eight relation/query tokens.  Its final answer is one
of eight separate answer tokens.

The eight structured relations include subject and object properties, modular
compositions of both entities, and conditional relations.  Six grammatical
templates are used for training; two unseen combinations of known markers and
word orders are held out.  Training uses next-token cross entropy across the
whole sequence, with the answer position upweighted but not isolated as the
only language-modeling target.

For a fraction rho of training facts, the structured answer is replaced by a
fixed balanced permutation.  Randomized sets are nested across rho.  Chance
coincidences are retained and recorded.  Held-out pairs and held-out entities
test structured generalization; arbitrary answers in those splits are negative
controls.

## Initial factorial and staging

Calibration first uses one data/initialization seed and conventional SwiGLU
models at depths 1, 2, and 4, width 96, with rho in {0, 0.5, 1}.  Two loads are
available; the smaller is used only if the larger cannot produce both a
high-fidelity arbitrary model and a capacity-pressure boundary.  Training is
adaptive to a fixed maximum step count and reports exposures per fact.

After calibration, compare conventional and bilinear/product MLPs only in
informative matched-fit cells.  Compare AdamW and plain SGD only after choosing
a load at which both can reach the fidelity gate.  No architecture or optimizer
conclusion may compare a fitted model with an underfit one.

Fidelity gate: at least 97% trained-template answer accuracy and 95% held-out-
template accuracy on the learned training facts.  Full-sequence loss is
reported separately and is not part of this gate.

## Regime coordinates

Each layer/condition is described by four independently measured coordinates:

1. **Axis privilege:** native-channel semantic prediction or sparse causal
   sufficiency minus matched random rotations.
2. **Semantic transfer:** intervention selected on one group of structured
   counterfactuals transfers the intended answer change to new entities and
   templates.
3. **Constraint strength:** normalized off-diagonal margin-gradient overlap and
   finite collateral after a unit self-margin parameter edit.
4. **Constraint breadth:** effective support of signed effects across facts.

Stable fact-pair decoding across templates is an address measure, not evidence
of semantic coherence.  Positive and negative activation tails and signed
causal effects are both retained.

Operational regime labels:

- neuron-aligned semantic: positive native-over-rotation axis privilege and
  successful single/few-channel semantic transfer;
- distributed semantic: semantic transfer succeeds but axis privilege does not;
- weakly coupled address memory: stable fact addressing, weak semantic
  transfer, low constraint strength, and broad support;
- taut constraint web: high constraint strength/collateral, broad signed
  support, low slack, and finite-edit prediction;
- underfit: factual gate fails, so no successful-memory regime is assigned.

Labels summarize measurements; borderline cells remain unlabeled.

## Primary falsifiers

- Neuron alignment is rejected if native axes do not beat 20 matched random
  rotations on held-out semantic prediction and top-k intervention transfer.
- A semantic subspace claim is rejected if a discovery-selected direction or
  subspace fails to transfer on disjoint entities/templates or is matched by
  label-shuffled controls.
- A constraint explanation is rejected at a tested edit scale if predicted
  off-diagonal finite effects have Pearson correlation below 0.5 or fail to
  beat source-permutation controls by 0.2.
- A taut-web claim is rejected if normalized coupling/collateral does not rise
  with pressure at matched accuracy, even if effective support rises.
- Any claimed fact or rule mechanism must be conditioned on achieved behavior;
  planted but unlearned facts are excluded.

## Red-team rounds

1. Generator leakage, exact-balance, split, intervention, and tiny-overfit
   checks.
2. Rotation, shuffled-label, mean-patch, output-token, shared-input, and
   competitor controls.
3. Larger panels and fresh data/initialization seeds for every decisive result.
4. Gauge/function-space audit for bilinear factors; raw factor weights are not
   compared before scale/sign/permutation canonicalization.
5. Natural-language bridge: rerun the cheapest validated diagnostics on the
   existing six-layer planted-fact LM and/or the 546M FineWeb checkpoint, while
   clearly separating observational natural facts from provably arbitrary
   planted facts.

All design changes after this lock are logged with the failed condition that
motivated them.  Negative and null results are retained.
