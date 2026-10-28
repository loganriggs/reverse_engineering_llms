## H761 — Fresh copying confirmation does not reproduce a reliable group advantage

A prospective panel is locked before outputs: eight color keys and eight tool/object values in four formats (semicolon, newline, colon and equals),32 pairs with three variants each. A task edit changes the repeated value; a nuisance edit renames the key while keeping the answer. All24 frozen representations from H759 and every seed are retained. No cases are excluded: all32 pairs are token-valid, and every format has100% both-endpoint choice,100% intended full-vocabulary top-1 and100% nuisance-choice correctness.

Head edits are separated into final-query position, the last occurrence of the copied value (source), and both. Only that last source occurrence is patched; this is not an intervention on every copy of the value. Endpoints are independently reconstructed and differenced, with the native baseline retained. No task training or seed selection occurs.

| H5 final-query task edit | Full effect RMS | Scalar error | Block error | Shared-group error | Group-null error | Fixed PCA error | Adaptive PCA error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Colon | 0.253 | 73.4% | 90.2% | 104.1% | 97.5% | 92.9% | 120.0% |
| Equals | 0.254 | 77.2% | 95.6% | 83.7% | 93.5% | 92.1% | 90.1% |
| Newline | 0.262 | 133.4% | 152.4% | 134.9% | 120.5% | 81.7% | 140.9% |

Values are mean seed-level relative RMS errors against full-head intervention, not activation reconstruction MSE. The semicolon cell has full effect RMS0.169, below the fixed0.25 reporting threshold; its scalar/group absolute errors are0.248/0.222logits. Even though the group mean is slightly better there, both miss much of the full effect. Fixed-rank PCA's relative advantage on newline also does not make its81.7% error good fidelity.

Source-only H5 task effects are much smaller: full RMS0.044–0.073 across formats. Both-position results largely track final-query results: scalar/group relative errors77.2%/109.5% for colon,74.8%/80.5% for equals,127.7%/127.1% for newline. The poor transfer is therefore not explained by overlooking a large effect at the tested source position.

H3 effects are weak throughout this copying panel: final-query task RMS0.023–0.064, source RMS0.006–0.021, both RMS0.033–0.067. They are retained with absolute errors, not used to manufacture large relative rankings. This constrains claims about which computations these heads perform at each position; it does not prove H3 is absent from the full copying circuit.

Nuisance fidelity still shows tradeoffs. For H5 newline, final-query full RMS0.315 yields scalar/group errors85.8%/107.8%; both-position full RMS0.344 yields74.4%/99.2%. Semicolon both-position nuisance errors are68.1%/101.0%. Colon/equals nuisance effects are smaller and remain reported absolutely. These are faithfulness measurements, not evidence that preserving every nuisance effect is desirable for task-specific abstraction.

Interpretation: the isolated H759 copying advantage does not establish transferable shared-group structure. The original model is competent on this fresh panel, but current eight-active-coordinate representations often poorly approximate the head's causal effect. No candidate qualifies as a reliable general featurizer here. This remains compatible with capacity limits, a mismatch between natural write-MSE and downstream sensitivity, weak coverage of repeated-copy contexts in training, or limitations of frozen coefficients/group gates. It does not justify adding hierarchy or semantic labels to the current groups.

Audits:18,528 forwards =96 native captures +18,432 interventions including identity controls. Manifest/model/representation hashes unchanged; identity logit error<=2.17e-5; full donor-head vector error<=1.77e-5; exact union error zero; outputs finite. No exclusions, no reserved-test evaluations; natural document pool26,112 unchanged. Native contexts and source positions are saved for reproducibility.

Next: distinguish capacity from objective mismatch. Evaluate fixed/adaptive rank or active-budget curves using the already frozen natural transforms, and compare with a task-blind downstream-sensitivity metric before retraining more complex groups. Keep the fresh panel as evaluation evidence rather than a source of membership labels. The separate explicit sentiment/capital wording test remains outstanding; H760's distinction between forced-choice information and unrestricted completion remains in force.
