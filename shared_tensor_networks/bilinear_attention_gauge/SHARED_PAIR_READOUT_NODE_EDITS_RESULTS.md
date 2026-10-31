# The missing operation is query-conditioned readout, not another head deletion

The previous [head](SHARED_PAIR_DATA_EQUIVALENT_NODE_EDITS_RESULTS.md) and [QK-branch](SHARED_PAIR_BRANCH_REALIZABLE_NODE_EDITS_RESULTS.md) tests asked whether physical attention edits reproduce deletion of the readable A–B, A–C, and B–C product nodes. A structural fact changes the interpretation: the trained model computes its two parity logits *only* from three primitive attention logits $b$ through a fixed quadratic readout,

$$y_p=b^\top Q_p b+\gamma_p.$$

An upstream edit that leaves $b$ exactly unchanged cannot change either parity logit. Thus zero primitive-logit collateral is impossible for an attention-side parity edit in this architecture. To look for a cleaner physical location, we fit edits to the actual symmetric $3\times3$ readout matrices $Q_p$. These leave all three primitive logits unchanged **by construction**. We compare a single edit shared across both query types, which is executable by changing the trained readout parameters, with a diagnostic that uses a different readout edit for each query type. The latter is *not* a native operation of this model.

For each of ten paired-token and ten dense-token trained models, fit the three strongest product-node deletion effects in the fixed weight-only rank-three joint-nullspace gauge. The data fit uses 20,000 seen-combination sentences, no answer labels, and tests on 30,000 independent sentences including the rare whole-combination holdout. A separate weight-only fit matches polynomial coefficients—constant, linear, and quadratic terms—over both query types without sampling sentences. The readout patch is checked against the model's direct forward pass, including its internal logit scaling.

| Coding and node | Native shared readout: seen-data test MSE | Native shared readout: weight-only test MSE | Query-conditioned diagnostic: seen-data test MSE | Query-conditioned diagnostic: exact coefficient MSE |
|---|---:|---:|---:|---:|
| Paired A–B | **.869** | 1.596 | **.0019** | .0035 |
| Paired A–C | .151 | .277 | .0006 | .0022 |
| Paired B–C | .079 | .145 | .0006 | .0016 |
| Dense A–B | **.920** | 1.654 | **.0005** | .0007 |
| Dense A–C | .132 | .235 | .0002 | .0005 |
| Dense B–C | .073 | .129 | .0002 | .0003 |

All errors are mean squared response error divided by mean squared target-node effect; 1.0 is no edit. The native readout edit can approximately realize A–C and B–C, more faithfully than whole-head or QK-to-constant edits, and with exactly zero change to primitive logits. It cannot realize A–B well with one query-independent matrix. On the rare combination its A–B residual MSE is 5.63 times full-test A–B effect energy in paired coding and 5.83 in dense coding. A weight-only native fit trades lower rare-combination error for worse ordinary-distribution error: the metric choice matters, but neither produces a faithful shared readout edit.

Allowing a *separate* readout edit per query collapses all six node/coding mean test errors below .002, including the rare holdout (A–B rare MSE is .036 paired and .011 dense relative to full-test target energy for the seen-data fit). The coefficient-space fit also becomes nearly exact over **all real count vectors**, not merely likely task sentences. This isolates the missing degree of freedom: a semantic product node corresponds closely to a readout quadratic **in a query-dependent representation of the three primitive logits**. A single parameter edit to the trained $Q_p$ is shared across queries and cannot select those different quadratic forms.

This is a strong mechanistic constraint, not a claim that the trained network has an explicit query-gated readout. The conditional edit is an oracle augmentation that diagnoses why the native edit fails. The model may implement the two query-dependent representations upstream, before their shared readout. The next analysis should compare the query-specific maps from native bit space to the canonical coordinates and test whether any stable within-head operation realizes them without merely moving the primitive logits. Because the readout depends only on those logits, any attention-side edit must accept and measure primitive-logit collateral; it cannot be optimized to zero while changing parity.

Reproduce with `shared_pair_readout_node_edits.py`; its JSON has all 60 fitted nodes, coefficient-space and task-distribution errors, rare-stratum results, fitted matrices, and direct-model edit checks. CPU only; no model retraining.
