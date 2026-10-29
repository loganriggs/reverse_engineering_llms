# M39: coactivation groups improve over random groups, but lose scalar causal fidelity

All 12 frozen M38 candidates evaluated on the unchanged native-intervention benchmark. Both external generators are applied identically within each seed. No fitting or candidate selection. This is familiar development evaluation, not semantic discovery.

Metric: native-probability-weighted squared log-probability response error, divided by native response energy. Lower is better; predicting no response scores 1. Responses are relative to each model’s own intact output.

| Seed | Source objective | Region | Scalar | Random groups | Coactivation groups | Learned/random |
|---|---|---|---:|---:|---:|---:|
| 71 | intact_KL | prefix | 1.2631 | 10.7135 | 5.8559 | 0.5466 |
| 71 | intact_KL | task | 0.7759 | 5.4614 | 3.0838 | 0.5646 |
| 71 | intact_KL_plus_native_response | prefix | 1.3166 | 7.0740 | 3.5879 | 0.5072 |
| 71 | intact_KL_plus_native_response | task | 0.9290 | 3.5378 | 2.3134 | 0.6539 |
| 72 | intact_KL | prefix | 1.1533 | 10.6661 | 2.8633 | 0.2684 |
| 72 | intact_KL | task | 0.6315 | 3.0722 | 2.4691 | 0.8037 |
| 72 | intact_KL_plus_native_response | prefix | 0.9645 | 7.5448 | 3.3390 | 0.4426 |
| 72 | intact_KL_plus_native_response | task | 0.5214 | 2.7497 | 1.8535 | 0.6741 |

Learned grouping reduces error versus matched random grouping in all eight comparisons (19.6–73.2%). However, every grouped candidate remains worse than its scalar source, and all eight learned-group errors exceed the zero-response baseline. This supports useful coactivation structure but does not validate faithful computational units. Scalar uses 32 active atoms; grouped unions use fewer on average, so this does not establish an architecture ranking at equal active cost.

The experiment applies external write perturbations. It does not remove group gates or test whether native channel groups substitute for one another. Those mechanistic questions remain open.

Audit: 4,828 native forwards; 4,224 response rows; 528 intact baselines; 704 original native directions including 96 no-ops. All 1,408 scalar-source response rows replay M37 exactly. Independent NumPy checks cover 144 sampled effects; all 1,008 summaries checked. Maximum sampled metric discrepancy 1.78e-15.

Next: full ordinary-output inventory for all candidates, then distinguish shared-atom removal from group-gate removal. Stop membership optimization on this panel; preserve whole-task exclusions. No manifold or new-task claim.

[Protocol](validation/coactivation_native_protocol.json) · [Results](validation/coactivation_native_fidelity.json) · [Audit](validation/coactivation_native_audit.json)
