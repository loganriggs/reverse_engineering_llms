# Weight-only query transport explains the readout-edit obstruction

The [readout-edit result](SHARED_PAIR_READOUT_NODE_EDITS_RESULTS.md) found that a *single* physical quadratic-readout patch fails to delete the readable A–B product, while an oracle with a different readout patch for each query nearly realizes all three semantic products. This experiment tests the proposed reason: the three primitive attention-logit coordinates rotate with the query. It uses the exact trained attention tables, with no sentence samples or task labels to select the map.

Let $H_0,H_1\in\mathbb R^{3\times8}$ map key-token counts to the three primitive logits under common and rare queries, before their shared read bias. Enumerate all six permutations $P$ of the primitive-logit rows. For each one, fit only three row gains $D$ by least squares and score $\|H_1-DPH_0\|_F^2/\|H_1\|_F^2$. The task's actual rare-query target order is B/C/A, corresponding to permutation (1,2,0), but that label is used only *after* selecting the best fit.

| Coding | Correct permutation recovered | Best weight-table relative MSE | Next-best permutation MSE | Held-out rare-combination primitive-signal MSE | Induced parity-coefficient MSE |
|---|---:|---:|---:|---:|---:|
| Paired tokens | **10/10** | .00405 | .631 | .00796 | .00952 |
| Dense overlapping tokens | **10/10** | .00075 | .635 | .00209 | .00180 |

The mean fitted gains are (.843, .818, .856) for paired tokens and (.819, .794, .809) for dense tokens. The held-out primitive-signal score uses 30,000 fresh task sentences per model, restricted at evaluation to the rare query and withheld whole combination; it is not used in fitting. The parity-coefficient score computes the quadratic forms $H_1^\top Q_pH_1$ versus $(DPH_0)^\top Q_p(DPH_0)$ for both parity outputs, over **all real key-count vectors**. Thus the learned query transport predicts more than sampled task accuracy.

The weight-only nullspace-gauge product nodes are strongly localized in query/output cells across all twenty models: A–B is largest in parity output 1 under rare query in 20/20 models, B–C in parity output 0 under common query in 20/20, and A–C in parity output 1 under common query in 20/20. The mean dominant-cell energy shares are .987/.997 for A–B, .995/.998 for B–C, and .680/.704 for A–C (paired/dense). A–C is spread across another cell, but its dominant cell is stable.

This makes the readout result intelligible. In native coordinates, both parity logits use fixed matrices $Q_p$ on the *current query's* primitive logits. In a token-semantic A/B/C frame, the rare query approximately applies $DP$ to those logits, so its effective quadratic form is conjugated by that transport. Deleting a fixed semantic product across queries therefore asks for **different edits to the shared native readout**. The oracle query-conditioned patch supplies exactly that missing degree of freedom; the trained model does not contain such a patch.

This is a meaningful weight-derived structure, but the toy training objective explicitly rotates the target bit order, so recovering the designed permutation is a validation of method rather than discovery of an unknown natural-language rule. For the broader canonicalization program, a single global sparse basis may be the wrong object when representations transform with context. A better candidate is a canonical feature bundle with learned transport maps between query conditions, tested on query families held out from fitting and on independent interventions. The physical implementation still needs separate evidence; the recovered map describes how the trained *function* moves between query conditions.

Reproduce with `shared_pair_query_transport.py`. The JSON includes every seed's selected and runner-up permutations, gains, exact table and parity-coefficient errors, rare-combination response error, and semantic-node cell energy. CPU only; no model training.
