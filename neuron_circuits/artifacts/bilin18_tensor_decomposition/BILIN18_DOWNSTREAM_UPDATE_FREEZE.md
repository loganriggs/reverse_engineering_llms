## H878 — Downstream MLP responses causally generate most early-reader context dependence under controlled readout

H877 localized the residual mismatch in an additive accounting. H878 intervenes on those computations: freeze selected final-token updates to the recipient's unedited native update while injecting the same positive/negative L8H3 headwrite. Other updates remain live. With shared final normalization and no logit cap, freezing all MLP updates from L8 through L17 reduces new-format C mismatch from 35.8% to 2.1% for country output and 27.1% to 3.1% for capital output. Response norms remain substantial, so convergence is not caused by erasing the headwrite's output effect.

### Intervention and controls

All 192 H873 contexts, including separately analyzed failed-capital D, receive six conditions: live, freeze L17 MLP, freeze L15–17 MLPs, freeze L8–17 MLPs, freeze L9–17 attention, and jointly freeze L8–17 MLP plus L9–17 attention. L8 attention retains the injected write in every condition. Freezes replace only the final query token's update with its own native value. All other tokens and unselected updates remain live. Identity arms run for every condition.

The joint condition leaves the injected write propagating only through residual multipliers. Its near-zero shared-scale uncapped cross-context mismatch is an architectural control, not a newly discovered invariant. This is an oracle intervention using cached native computations, not a learned featurizer or an efficient replacement model. Reused country contrasts and format C remain posthoc diagnostic evidence; there are no new semantic tasks or reserved-test evaluations.

### Competent format C, shared final scale and uncapped readout

Response norm ratios compare each intervention with the live response across the same 12 canonically oriented pair means. A norm ratio measures strength, not agreement of every pair response.

|Output|Frozen updates|Relative mismatch|Absolute mismatch RMSE|Copy response norm / live|Capital-context response norm / live|
|---|---|---:|---:|---:|---:|
|country|live|0.3578|0.6503|1.000|1.000|
|country|mlp17|0.2945|0.5441|0.936|1.017|
|country|mlp15_17|0.2371|0.4322|0.858|1.003|
|country|mlp8_17|0.0211|0.0398|0.877|1.037|
|country|attention9_17|0.2413|0.5999|1.280|1.368|
|country|joint|0.0000|0.0000|0.898|1.070|
|capital|live|0.2713|0.4525|1.000|1.000|
|capital|mlp17|0.2222|0.3582|0.939|0.966|
|capital|mlp15_17|0.1795|0.2684|0.857|0.896|
|capital|mlp8_17|0.0311|0.0407|0.837|0.784|
|capital|attention9_17|0.2262|0.4979|1.288|1.320|
|capital|joint|0.0000|0.0000|0.868|0.819|

Freezing only L17 or the last three MLPs gives partial improvements. Freezing all downstream MLPs gives the large reduction, consistent with distributed accumulation. Freezing later attention alone leaves 22.6–24.1% relative mismatch; its absolute mismatch is 0.498–0.600, compared with live 0.452–0.650. Its lower relative error therefore partly reflects larger response magnitudes, and is not a comparable absolute convergence.

With native normalization and logit cap, freezing all downstream MLPs reduces country mismatch 28.0%→19.0% (absolute RMSE 0.539→0.370) and capital mismatch 36.1%→9.0% (0.630→0.114). Final readout effects still matter. The shared-scale result must not be presented as the original model becoming universally context-invariant.

### What this establishes, and what it leaves open

Downstream MLP responses are a causal contributor to the measured context dependence. The same input write can be transformed differently depending on recipient context, while much of the direct output response remains reusable. The result does not establish that attention is causally irrelevant: attention can shape inputs to MLPs, and unselected attention changes in the MLP-freeze condition remain live. Nor does it identify a unique semantic feature, validate a manifold, or demonstrate task accuracy under a useful replacement model.

The next distinction is between context-specific local transformations and nonlinear changes over the finite perturbation. A recipient-native linearization preserves each MLP's local Jacobian while removing its higher-order response. Compare live MLPs, recipient-native linearized MLPs, and frozen MLPs across a small predeclared headwrite-amplitude ladder. If linearization preserves the mismatch, different local Jacobians are enough; if mismatch emerges primarily with finite amplitude and disappears under linearization, nonlinear traversal matters. Include RMS normalization in the MLP map being linearized, keep attention live, and audit identity plus numerical derivatives before interpretation. This remains a mechanistic diagnostic preceding broader task transfer and featurizer validation.

### Verification and provenance

Completed 3,648 full-model forwards. Every freeze identity reproduces the native final state exactly; live arms reproduce H876 exactly. Joint-freeze skip prediction has relative vector error at most 2.9e-05 (maximum coordinate error 0.0123, retained because vectors are large). Selected native readout matches full-model logits within 8.4e-06. H876 live response RMSE replays within 0. Source hashes unchanged; all outputs finite. Pair-bootstrap intervals (4,096 samples, uncorrected) are saved in the analysis JSON, separately A/B, C and failed D. The natural document pool remains 26,112 and the broad goal remains active.
