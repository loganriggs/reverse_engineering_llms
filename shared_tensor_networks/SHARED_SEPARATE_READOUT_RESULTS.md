# Shared versus independent task dictionaries

300new independent-head fits, grouped into30models, compared with15archived shared models. Five d8/h4teachers, three paired initializations each, ten supervised tasks,512common input examples and5120source scalar labels, noise.05. Protocol fixed in SHARED_SEPARATE_READOUT_PLAN.md. These teachers were already used in scaling; this is a mechanistic control, not a new independent teacher confirmation.

Sharedh4has164parameters. Ten independenth4heads have740parameters and start with exactly the same source predictions as the shared model. Ten independenth1heads have170parameters, the nearest integer uniform-width budget; their initial function and capacity differ. Each independent head has its own200evaluationTRFcap. Independent optimization changes the trust-region schedule and total computation even though the objective is separable. This does not isolate tying alone or provide a compute-matched optimizer comparison.

All methods adapt through exactly ten predicted source outputs, normalized to unit GaussianRMS. Independent models do not receive100hidden-product transfer features. Four whole held-out heads receive the same32/128noisy labels; population projection floors and oracle source-output features are retained. Unit population normalization is privileged synthetic information but uses no held-out labels.

| Method | Parameters | Median teacher source MSE | Median teacher transfer MSE,128labels (range) | Median population transfer floor |
|---|---:|---:|---:|---:|
| shared | 164 | 0.00020528 | 0.000851505 (0.000462128–0.0122857) | 7.52507e-05 |
| separate_width | 740 | 0.0313684 | 0.109226 (0.0106614–0.215404) | 0.086619 |
| separate_budget | 170 | 0.470332 | 0.336919 (0.0810686–0.457819) | 0.23402 |

Entries first average four adaptation tasks and three initialization outcomes within each teacher. No best-head or best-seed selection.

| Teacher | Shared transfer | Independenth4transfer | Independenth1transfer |
|---|---:|---:|---:|
| 225001 | 0.000462128 | 0.206812 | 0.336919 |
| 225002 | 0.000597858 | 0.109226 | 0.457819 |
| 225003 | 0.000851505 | 0.215404 | 0.392914 |
| 225004 | 0.0122857 | 0.0502137 | 0.277404 |
| 225005 | 0.00492277 | 0.0106614 | 0.0810686 |

Shared beats independenth4in5/5teacher blocks and independenth1in5/5. These counts apply to the prescribed output-span transfer endpoint, not every possible use of the learned representations.

## What this control can and cannot establish

There is direct evidence of optimization failure in13/15independent equal-width fits: copying the corresponding learned shared dictionary into each independent head gives a strictly lower empirical loss. check_shared_solution_embedding.py verifies all copied predictions and loss values. With identical data, normalization and no added parameter penalty, the independent hypothesis class contains the shared class, so its global minimum training loss cannot be higher. These fitted independent solutions are therefore demonstrably not global empirical minima. This does not prove all of the transfer gap is optimization, nor establish the shared solutions as global minima.

The comparison measures the benefit of fitting a common dictionary jointly versus fitting each task independently, under the implemented optimization protocols. The wider independent model has more parameters and may overfit or optimize differently. The near-budget model is much more restricted per task. Shared-versus-separate differences cannot be assigned uniquely to parameter count, regularization, basin geometry or reuse without further controls.

Population projection floors distinguish an inadequate span of learned source functions from finite adaptation-label noise. Low training-task error or a strong output span is not itself evidence that hidden components match the teacher. The shared models already have separate component-recovery measurements; this experiment does not identify a physical execution DAG.

Independent audits verify provenance against the prior scaling audit, every initial/final independent-head derivative against direct autograd, coefficient predictions on400fresh inputs, all source losses, normalization, and transfer least-squares normal equations and exact Gaussian errors. All45groups and300individual head fits are retained; maximum relative Jacobian discrepancy1.03e-15. Source hashes and every head status/evaluation count are saved.

Decision: close this fixed comparison. Use the hypothesis ledger to separate statistical sharing effects from optimization dynamics; do not rescue independent or shared fits with selected retries.
