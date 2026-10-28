## H802 — Transferring the earliest divergent attention responses leaves most of the behavioral gap intact

H801 found the first context-dependent response difference in L3 attention, the fourth block. To test whether that early difference explains the behavioral context gap, preserve each recipient hybrid baseline and replace its perturbed L3 attention output with baseline_recipient + (perturbed_other_context - baseline_other_context), matched by lexical item, format, query choice, and transplanted L2 source state. Swap at source2, final query, or both; retain a no-transfer arm. Run delta=0 and +.50 for every arm, so each response has its own baseline.

### Results

The context gap is the different-key minus same-key change in intended-token log probability caused by the fixed value perturbation. Results average over both local-state choices and queries within lexical item.

| Transfer site at L3 attention | Vertical-bar context gap | Colon/newline context gap |
|---|---:|---:|
| None | .3862 | .2172 |
| Source2 | .3829 | .2071 |
| Query | .3811 | .2184 |
| Both | .3778 | .2082 |

For joint transfer, the paired change in gap is -.00836 with descriptive item-bootstrap interval [-.02432,.00644] in vertical bars and -.00901 [-.01783,-.00140] in colon/newline. Thus the early responses have, at most, modest average leverage on this contrast under the tested transfer. This is not an equivalence claim or a unique mediation fraction. Source-only transfer in colon/newline also modestly reduces the gap (-.01017 [-.02021,-.00147]); query-only effects are near zero on average.

The colon/newline mean perturbation response remains positive in different-key contexts (+.1001) and negative in same-key contexts (-.1081) after joint transfer. The vertical-bar means remain -.0617 and -.4395. Pooled full-vocabulary accuracy remains 98.4% and 100%, respectively, in every arm.

### What this changes

A large or early vector discrepancy is not sufficient evidence that it explains the output effect. This test weakens the hypothesis that the two earliest divergent attention responses are the main transferable cause of the gap. It does not show that L3 is irrelevant: unpatched token positions, local MLP responses, later paths, and interactions with baseline states remain. The recipient background is deliberately retained, so transferred response vectors can be interpreted differently downstream.

Next refresh the competing hypotheses, then use broader residual-response transfers at selected depth cuts. Include all token positions versus query-only transfer to distinguish alternative routes from later conditioning. At the final residual, a response transfer tests whether normalization/readout against different baseline states can itself preserve the behavioral gap. Intermediate cuts can test whether downstream computation regenerates context dependence. Fresh lexical and diverse competent-task validation remain necessary before treating this as a reusable computational unit or featurizer target beyond this panel.

### Verification and limits

1768 model forwards: 104 native contexts plus 1664 crossed-state/delta/transfer runs. Tracked model, manifest, source capture, trace capture, reference, and runner hashes unchanged. Projected direction mismatch zero; local residual clamp relative error 1.40e-8. Native-state zero-delta logits match within 2.10e-5. The no-transfer response reproduces H800 within 5.33e-6 across both local-state conditions. All transfer-arm baseline intended log probabilities match the no-transfer baselines within 3.19e-6, and baseline KL values within 1.20e-7. These replay controls check that zero-response transfer is inert to numerical precision. H799 native-state response replay is within 4.93e-6; all response values finite.

The 16/10 aligned lexical items and two prompt formats are reused. Bootstrap intervals average duplicate/query and local-state variants within item, are descriptive, and have no multiplicity correction. Hybrids need not lie on the native manifold. The original reserved test remains untouched; natural-document pool remains 26,112. This experiment validates a controlled intervention implementation, not a new featurizer, manifold, semantic equality gate, or full computational DAG.
