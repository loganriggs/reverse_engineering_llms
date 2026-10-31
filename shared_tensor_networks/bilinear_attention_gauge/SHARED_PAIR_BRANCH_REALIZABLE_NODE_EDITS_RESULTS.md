# Can QK-branch edits realize the readable product nodes?

The previous [head-edit comparison](SHARED_PAIR_DATA_EQUIVALENT_NODE_EDITS_RESULTS.md) showed that semantic product-node deletion is poorly reproduced by attenuating whole physical heads or reweighting their six quadratic head-pair terms. A head is a coarse unit. Here the candidate intervention is finer and executable: for each of three real attention heads, independently move either QK score toward the constant score 1. The six doses are bounded between 0 and 1; the resulting branch products and downstream quadratic readout are evaluated **exactly**, including interactions between simultaneous edits. The table algebra is checked against the existing branch-intervention implementation at a nontrivial dose.

For each of ten paired-token and ten dense-token trained models, fit the six doses on 10,000 independent sentences that exclude the rare whole-combination holdout. The target is deletion of each of the three strongest product nodes in the weight-only, rank-three joint-nullspace gauge. Fit only parity-output changes, without answer labels; evaluate on 20,000 new sentences including the rare combination. A no-edit baseline has relative effect MSE 1.0.

| Coding and semantic product | Full-dose test relative parity-effect MSE | Rare-combination MSE / full-test target energy | Quarter-dose relative MSE |
|---|---:|---:|---:|
| Paired A–B | **.807** | 4.953 | .811 |
| Paired A–C | .481 | .422 | .740 |
| Paired B–C | .600 | .292 | .629 |
| Dense A–B | **.963** | 6.870 | .962 |
| Dense A–C | .562 | .424 | .605 |
| Dense B–C | .573 | .306 | .628 |

The A–B product remains the hardest: the fitted branch doses are generally small, especially in dense-token models, so the edit barely approaches the semantic counterfactual. A–C and B–C can be partly matched at the fitted strength, but their effect error remains substantial; for paired A–C, it rises from .481 at full strength to .740 when both the physical doses and desired semantic deletion are quartered. This is a held-out *edit-strength* test, not just a new-sentence test. Rare-combination errors are normalized by the full-test node-effect energy because the node's own energy on a particular rare subset may be nearly zero.

These QK-to-constant edits also change the three primitive bit logits. The JSON reports their RMS change alongside the parity target RMS so a parity fit is not mistaken for an isolated computation edit. The experiment does not search arbitrary within-head directions, context-dependent patches, or all possible physical graph factorizations. It establishes that this specific six-control branch family is insufficient to realize the three readable product nodes. The next candidate family should use within-head value/QK subspaces and explicitly penalize collateral primitive-logit changes, while keeping whole-combination and dose holdouts. If those candidates also fail, the product graph should remain a **function-level** explanation, not a claimed physical implementation.

Reproduce with `shared_pair_branch_realizable_node_edits.py`. Its JSON includes all 60 fitted edits, test strata, three strengths, fitted doses, and collateral bit-logit changes. CPU only; no new model training.
