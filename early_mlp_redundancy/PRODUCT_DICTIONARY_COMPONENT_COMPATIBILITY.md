# Average training gradients do not explain the source/final transfer conflict

M99 tested the simplest local objective-conflict explanation of M98: perhaps improving source interventions necessarily worsens final-position interventions in the fitted coefficient space. The measured population gradients oppose that explanation at all four tested states.

We held the model fixed at the two original initial states and the two full-pool KL-only final states. Each measurement covers all 140 training endpoints and the exact 70 natural-document assignments, with repeated natural documents evaluated once and weighted by multiplicity. No held-out gradients or parameter updates were used.

## The prediction being tested

For losses Li and Lj with coefficient gradients gi and gj, an infinitesimal Euclidean step −eta gj changes Li to first order by −eta gi·gj. A negative dot product therefore indicates local opposition; a positive dot product indicates a direction that improves both. This statement concerns the chosen coefficient coordinates. It is not invariant to arbitrary reparameterization, does not describe Adam's preconditioning/history, and does not guarantee finite-step or held-out improvement.

The source and final losses are their exact contributions to the balanced causal objective. We preserve each endpoint's divisor: six nonidentity branches when source and final interventions are available, three when only final interventions are available. The combined training objective weights natural and task updates equally:

L = 0.5 K_natural + 0.5 K_task + 0.5 lambda (B_source + B_final), with lambda = 0.18553117636658983.

## Results

| Seed and state | cos(source, final) | cos(source, KL objective) | cos(final, KL objective) |
|---|---:|---:|---:|
| 71 initial | +0.1347 | +0.2790 | +0.4519 |
| 72 initial | +0.1758 | +0.3137 | +0.3539 |
| 71 after KL training | +0.1053 | +0.2413 | +0.4170 |
| 72 after KL training | +0.1222 | +0.2553 | +0.2632 |

All four source/final dot products are positive. The negative gradient of the combined causal loss, the KL objective, and the full joint objective is predicted to improve both source and final training losses at every tested point. These signs agree when the entire population is recomputed by differentiating grouped source/final losses directly.

This is not universal alignment of every component. Among the six individual causal corners, four of fifteen unordered pairs have slightly negative cosine at each state; the most negative is between −0.0032 and −0.0038. Aggregate compatibility can coexist with small component-level tensions or unmeasured example-specific conflicts.

## What this means for the research direction

M98's held-out source regressions cannot simply be identified with opposing average training gradients at these states. The training objectives admit a shared local descent direction. This is compatible with M96's training gains alongside same-family transfer reversals, and M97's observation that held-out regressions were already locally predictable along the actual KL-to-joint endpoint direction.

The distinction is consequential: **jointly improvable training behavior is not evidence of a transferable computational unit.** A featurizer can satisfy several causal labels on its training contexts without capturing the native rule that generalizes across contexts. Broader coverage helped in M98, but did not establish that rule.

We have not identified the cause uniquely. Population averages can hide example-level conflict; actual alternating Adam updates can differ from population Euclidean descent; finite steps and other states may behave differently. This result does not prove that more training succeeds, that capacity is sufficient, or that source/final objectives never conflict. It provides no evidence for native-channel cancellation.

## Verification

Four fixed preflight cases cover both seeds and source-eligible/ineligible endpoints. Scalar decomposition and gradient linearity passed thresholds frozen before execution. Their relative gradient discrepancies were approximately 2.3e-6 to 9.7e-6.

The population producer and full grouped-gradient recomputation each used 4006 native forwards. The latter independently accumulates with NumPy and differentiates source/final sums as grouped losses rather than adding separately differentiated corner gradients. It verifies every endpoint's scalar replay, schedule and divisor. Maximum relative gradient disagreement was 4.58e-6; all 80 reported aggregate directional signs agree. Both use the same native Transformer and autograd implementation, so this is not an independent Jacobian implementation.

All four states, 560 endpoint records, eight primitive gradients per state and complete Gram matrices are retained. No model selection, objective reweighting or optimizer search was performed.

M99 is closed. The next conceptual step is to investigate shared versus context-specific native operators, using the exact bilinear midpoint identity and training-only factorization, before interpreting replacement coordinates as computational units. First inspect the existing operator caches and prior tests to avoid repeating the local-chart experiments. A bounded native-operator test should deliver a candidate shared/private basis for featurizers, with separate held-out finite-change and downstream-causal validation. It should not reopen an unrestricted optimization detour.
