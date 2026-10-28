## H801 — Context-dependent perturbation trajectories diverge in the next attention block

Following H800, trace each +.50 value perturbation minus its own zero-delta hybrid baseline. Cross the surrounding same/different-key context with each fixed transplanted local source state. Capture attention output, MLP output, and block residual at the first source, second source, and final query through zero-indexed layers L2–L17. This repeats 104 native contexts and runs 416 hybrids, totaling 520 forwards.

At the intervention block L2, source residual response vectors agree across contexts to relative RMS error 5.3e-7 in both formats. The query response is exactly zero. The earlier first-source response is exactly zero throughout all layers, providing a causal-order control.

The next attention block L3 is the first available cross-token computation after the injection. Context-dependent response differences appear there at both source2 and query. Relative differences divide the RMS norm of the paired response difference by the RMS response norm pooled over the two contexts:

| Format | L3 attention source2 | L3 attention query | L3 block residual query |
|---|---:|---:|---:|
| Vertical bars | .297 | .371 | .522 |
| Colon/newline | .353 | .349 | .443 |

These are vector discrepancies, not fractions of causal effect. L3 query-attention response cosine is still .936/.943 across contexts. By L8, query-attention relative discrepancy is 1.059/1.006 and cosine .469/.512. By L17 the query residual discrepancy is .883/.663. Thus differences are visible internally before final output normalization, but their existence does not establish their contribution to the behavioral gap or eliminate later readout contributions.

This trace nominates L3 attention response at source2 and query for a paired causal transfer test. Replace the recipient perturbed attention output by its own baseline output plus the other context's perturbation response, keeping local-state identity fixed. Test source and query separately and jointly, with zero-delta baseline controls. The earliest divergence is structurally expected; causal tests must establish whether it matters rather than interpreting a large internal norm as importance.

Audits: model/input/reference/runner hashes unchanged, direction mismatch zero, local residual clamp relative error 1.40e-8, native zero-delta full-logit replay 2.05e-5, H799 response replay 4.93e-6, all measured response norms finite. The original reserved test remains untouched and the natural-document pool remains 26,112. Analyses pool query choices and both local-state conditions; duplicate prompts and shared lexical items are not independent samples. This is a reused diagnostic panel, not fresh validation. No featurizer/manifold/DAG is validated by this trace.
