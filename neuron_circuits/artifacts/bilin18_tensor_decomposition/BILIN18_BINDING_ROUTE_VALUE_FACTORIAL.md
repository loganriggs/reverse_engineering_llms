## H918 — Binding effects at L14H4 are carried predominantly by the persistent value stream

The previous turn completed H917 and changed the next experimental priority. H918 returns to the existing competent binding and matched copying panel. It tests all eight name pairs, both templates and both endpoints. First four pairs remain discovery; last four are transfer. These transfer names have appeared in earlier experiments and are not a pristine test set. No features or heads are selected using these results: the heads are the previously specified L8H3, L13H0 and L14H4.

### Intervention and implementation

For recipient final-token routing A and mixed values V, head output is sum(A*V). The donor is the opposite source-name endpoint in the same pair, template and family. Routing-only uses donor routing with recipient values. Mixed-values-only uses recipient routing with donor values. Bus-only changes only the persistent layer-0 value contribution at this reader; current-values-only changes the remaining value contribution. Full replacement uses the cached donor head output. All edits are applied before the head output projection; all other heads and downstream computation run live.

The model's actual mixing coefficient is used without an extra amplitude rescaling. The bus intervention changes this local reader's consumption, not the persistent stream globally. Identity deltas are zero. Exact local decomposition includes the routing-by-value cross term; downstream answer effects need not add linearly.

### Transfer results

Positive margin drop means reduced preference for the recipient's correct answer over the alternative name. Baseline task top1 is 87.5%; copying is 100%. Each row contains 16 endpoints across four transfer pairs and two templates.

|Family|Head|Swap|Mean margin drop|RMS change|Native→edited KL|Top1|
|---|---|---|---:|---:|---:|---:|
|copy|L8H3|routing|0.002|0.137|0.0014|100.00%|
|copy|L8H3|mixed_values|1.885|1.939|0.0661|100.00%|
|copy|L8H3|bus|1.858|1.911|0.0636|100.00%|
|copy|L8H3|current_values|0.014|0.022|0.0001|100.00%|
|copy|L8H3|full|1.892|1.962|0.0699|100.00%|
|copy|L13H0|routing|0.002|0.060|0.0002|100.00%|
|copy|L13H0|mixed_values|0.767|0.787|0.0099|100.00%|
|copy|L13H0|bus|0.730|0.749|0.0090|100.00%|
|copy|L13H0|current_values|0.032|0.033|0.0000|100.00%|
|copy|L13H0|full|0.784|0.810|0.0098|100.00%|
|copy|L14H4|routing|0.008|0.399|0.0141|100.00%|
|copy|L14H4|mixed_values|1.537|1.640|0.0559|100.00%|
|copy|L14H4|bus|1.519|1.619|0.0541|100.00%|
|copy|L14H4|current_values|0.018|0.023|0.0000|100.00%|
|copy|L14H4|full|1.581|1.621|0.0515|100.00%|
|task|L8H3|routing|0.008|0.065|0.0002|87.50%|
|task|L8H3|mixed_values|0.448|0.459|0.0053|87.50%|
|task|L8H3|bus|0.440|0.452|0.0051|87.50%|
|task|L8H3|current_values|0.008|0.010|0.0000|87.50%|
|task|L8H3|full|0.456|0.465|0.0063|87.50%|
|task|L13H0|routing|0.063|0.077|0.0001|87.50%|
|task|L13H0|mixed_values|0.773|0.793|0.0131|81.25%|
|task|L13H0|bus|0.759|0.778|0.0125|81.25%|
|task|L13H0|current_values|0.014|0.018|0.0000|87.50%|
|task|L13H0|full|0.834|0.855|0.0155|81.25%|
|task|L14H4|routing|0.236|0.283|0.0019|87.50%|
|task|L14H4|mixed_values|3.400|3.416|0.2530|43.75%|
|task|L14H4|bus|3.389|3.406|0.2509|43.75%|
|task|L14H4|current_values|0.010|0.027|0.0001|87.50%|
|task|L14H4|full|3.639|3.650|0.2953|37.50%|

The strongest binding effect among these frozen readers is L14H4: full replacement drops top1 to 37.5%; mixed-value and bus-only swaps both reach 43.75%. Bus-only mean margin drop is 3.3895 versus 3.3999 for all mixed values; current-values-only is 0.0101. At L13H0 and L8H3, bus contributions also closely track mixed-value effects, but binding effects are smaller. Copying shows substantial margin changes without top1 failure, illustrating why accuracy alone is insufficient.

Routing-only changes are smaller in this endpoint comparison. This does not establish that routing is unnecessary: native routing remains essential to reading values, these endpoints may induce similar routing, and only three heads are tested. The result localizes a causal carrier; it does not prove a manifold, a binding-specific feature, or completeness of the circuit.

### Next information-gaining test

Fit uncentered paired-difference subspaces of the persistent head-specific name values using discovery pairs only, with ranks 1, 2 and 4 plus matched random controls. Replace the full donor bus difference by its projection, preserving the recipient's actual routing and downstream computation. Evaluate on transfer names in both binding templates and both copying controls. Compare projected intervention effects with the full bus intervention using answer-effect error and KL to the full edited distribution. Report captured value energy separately. The critical possibility is that a shared lexical carrier works for both tasks while context-dependent routing/readout determines its use. Failure of a small subspace would instead favor high-dimensional lexical identity or insufficient discovery coverage.

### Verification

1,216 full-model forwards. Identity replay exactly zero. Native cached-logit replay max error 1.24e-5. Cached signed attention-times-value reconstruction relative error 4.89e-7. Routing/value/cross decomposition and bus/current decomposition max absolute error 3.29e-14. Source hashes unchanged, all measurements finite. Raw endpoint metrics and full edited logits saved. No reserved-test evaluation; natural document pool unchanged at 26,112. Broad goal active.
