# Does the joint sparse dictionary describe the trained heads?

The [joint-function toy result](JOINT_FUNCTION_TOY_RESULTS.md) found two label-free sparse key directions that align with the task's A and B bits. This experiment asks a narrower physical question: can that **fixed two-direction dictionary** represent each actual head of the ten trained models, and does it survive branch-level interventions?

For each seed, take the dictionary \(D\in\mathbb R^{2\times6}\) chosen from the combined original token function. For each physical head table \(H_h\in\mathbb R^{4\times6}\), fit only head-specific coefficients \(L_h\in\mathbb R^{4\times2}\) by least squares, so \(\widehat H_h=L_hD\). No task labels, intervention responses, or head-ablation outputs enter the dictionary selection or this coefficient fit. Compare 128 random two-dimensional key subspaces per seed. All numbers below average over 20 heads.

| Test | Mean relative MSE |
|---|---:|
| Original head table, fixed joint dictionary | **.00469** |
| Original head table, random dictionary mean | .664 |
| Original head table, random dictionary fifth percentile | .253 |
| Deleted-head logit response, fixed dictionary | **.00497** |
| QK1 branch replaced by constant, fixed dictionary | **.465** |
| QK2 branch replaced by constant, fixed dictionary | **.574** |
| Either branch replaced by constant, freshly fitted rank-two SVD | **below 1e-30** |

The deleted-head comparison is checked against the direct trained model's head outputs and agrees exactly for the original tables before approximation. Its low error is mainly a consequence of reconstructing the original head table: deletion is linear in that table. **It is not independent evidence that a component corresponds to a physical node.** The fixed dictionary's two-dimensional *span* is identical under SVD and sparse rotation, so this reconstruction test validates the joint low-rank key space, not the particular sparse orientation or its A/B names.

Nor do the physical heads separate neatly into A and B. Across the 20 original head tables, the A-bit fraction of A+B bit-flip energy ranges from **.287 to .661**; no head exceeds .8 or falls below .2. Each head mixes the two task variables, while the effective function admits a sparse semantic basis. This is precisely why a component of a canonical **function** need not be a literal head.

The branch interventions are decisive about scope. The dictionary was selected on the original function; it was not refitted after replacing QK1 or QK2 scores by one. Each intervened head table still has exact rank at most two in this key unfolding, so the large fixed-dictionary errors are **basis-transfer failures**, not insufficient rank. The intervention can reveal key directions suppressed or cancelled in the original function. A causal canonicalizer intended to explain physical computations must either include these interventions in its fitting objective or explicitly abstain from predicting them. A function-only canonicalizer remains useful for compactly describing ordinary token behavior.

Reproduce with `dual_joint_physical_redteam.py` after generating `dual_joint_function_canonical.json`. `dual_joint_physical_redteam.json` records all twenty heads, per-head random controls, exact head-deletion checks, bit-flip energies, and both branch-intervention errors. This is a six-key toy and a full role-valid token table; the result does not establish the same low-rank semantic space in the 546M model.
