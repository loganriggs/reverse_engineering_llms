# Isolating the extra variable-projection curvature

20fits on the same five archived teachers and two starts, identical reduced objective, unit-factor/radial scale control, readout ridge .001, hidden-step damping .001, Armijo schedule and100curvature-call cap. Only curvature differs: full JredᵀJred=AᵀA+BᵀB versus projected AᵀA. Same gradient. All readouts refitted exactly. This is not the earlier scipy TRF algorithm or compute budget.

| Teacher | Full source MSE | Projected source MSE | Full transfer MSE | Projected transfer MSE |
|---|---:|---:|---:|---:|
| 232101 | 0.000141611 | 0.000141611 | 0.000373547 | 0.000373547 |
| 232102 | 0.0288452 | 0.000116002 | 0.0403223 | 0.00028748 |
| 232103 | 9.55838e-05 | 0.00558954 | 0.0003724 | 0.00760699 |
| 232104 | 0.0179209 | 0.00443831 | 0.0246965 | 0.0120198 |
| 232105 | 0.0170398 | 0.000167897 | 0.0422912 | 0.00058423 |

Each entry averages two starts; transfer averages four whole-task holdouts with128noisy adaptation labels each. Source and transfer errors use exact Gaussian coefficient moments. No held-out data enter source updates.

Projected curvature improves terminal source objective in4/10pairs, worsens2/10, and matches4/10within absolute1e-6. It reaches all-four-feature matched cosine>.99 in8/10runs versus7/10full. This threshold is not equivalent to low transfer error: a nearly recovered representation can still incur substantial errors on sensitive tasks. Preserve both metrics.

Projected improves both starts on teacher232102, first start on232104, and second start on232105. It worsens second starts on232103and232104. All runs stop on relative loss change except one full-curvature232102/233001run which reaches100calls; no line search exhausts. A capped run is not evidence of an unrepresentable function or a local minimum.

## Audit

All20trajectories and654attempted/accepted steps checked: independent literal-forward/QR objective replay, stored direction equations, every rejected/accepted Armijo decision, and exact parameter transitions. Independent autograd envelope gradients pass at every final checkpoint. Final exact Gaussian source and transfer errors replay; transfer fitted predictions agree with independent QR. Checkpoint and source-data hashes checked. The initial curvature identity has a separate10fixture verification from the preceding stage.

## What this establishes

Changing the extra positive-semidefinite term alone is sufficient to change finite-budget outcomes under this fixed damping and line-search rule. Removing it helps several previously difficult cases, but produces new failures elsewhere. Therefore this term is a real part of the learning-dynamics explanation, not a universal obstacle that should always be removed. This experiment does not retroactively attribute every previous TRF difference to that term: TRF adapts its trust region differently.

## A local descent theorem, and its limit

At the same state, set H=AᵀA+lambda I with lambda>0 and E=BᵀB positive semidefinite. Full step is −(H+E)^{-1}g; projected step is −H^{-1}g. Inverse order gives

 gᵀ H^{-1} g >= gᵀ(H+E)^{-1}g.

Proof: conjugate by H^{-1/2}; (I+H^{-1/2}EH^{-1/2})^{-1} has eigenvalues in(0,1]. Thus projected curvature gives at least as much predicted infinitesimal objective descent before line-search scaling. This does not order step norms, accepted decreases, nonlinear trajectories, feature recovery or generalization. All ten paired initial directions satisfy the inequality numerically; later states differ and cannot be compared as a common-state theorem.

## Decision

Close this bounded curvature-mechanism panel. We now have a verified implementation, an exact geometric distinction, and a controlled empirical intervention with positive and negative cases. Do not pursue another rescue sweep to turn8/10into10/10. Next synthesize the overall evidence for learnability and recoverability and audit remaining scaling/identifiability requirements, keeping ordinary joint optimization as a required baseline for future scientific comparisons. The broad research goal is not yet marked complete.
