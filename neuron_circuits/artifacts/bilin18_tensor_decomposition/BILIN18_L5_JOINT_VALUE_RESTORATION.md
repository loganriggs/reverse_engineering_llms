## H783 — Joint restorations reveal interacting contributions to the copied-value pathway

The question was whether the early/late MLP effects identified by individual restorations combine independently. We jointly restored receiver module outputs during a donor run, at both copied-value source positions, then measured only the resulting L5H5 local-value change in a fresh receiver run. This two-pass intervention excludes other downstream paths. Task edits change copied values; nuisance edits rename keys while preserving the requested answer.

The measured effect is the change in the answer logit margin, summarized as root mean square (RMS) across edits. It is a selected pathway's causal readout, not whole-model accuracy or a fraction of computation explained.

| Receiver outputs restored in donor run | Task surviving margin RMS | Key-renaming surviving margin RMS |
|---|---:|---:|
| None: complete target | 0.2692 | 0.1723 |
| L1 MLP | 0.1154 | 0.1378 |
| L4 MLP | 0.1653 | 0.0875 |
| L1 + L4 MLP | 0.0843 | 0.0706 |
| L3 + L4 MLP | 0.0997 | 0.0614 |
| L1 + L3 + L4 MLP | 0.0714 | 0.0592 |
| All five MLPs | 0.0665 | 0.0566 |
| All five attention outputs | 0.7490 | 0.0000 |
| All ten outputs | 0.0609 | 0.0000 |

Joint MLP restorations suppress more of this readout than either selected individual restoration. The selected three MLPs approach the all-MLP result on this reused panel. This prioritizes those sites for targeted follow-up, but does not establish a minimal circuit or a transferable feature representation.

The contributions are strongly non-additive. Predicting a joint restoration by subtracting the sum of individually removed vectors from the target fails even before the downstream model: the L1+L4 task vector discrepancy is 42.9% of target vector RMS; for all MLPs it is 197.8%. The corresponding margin discrepancies are 0.1370 and 0.5653. Thus assigning independent causal shares to those MLPs would be misleading. These comparisons diagnose context-dependent upstream transformations; they do not identify which nonlinearity is responsible.

Restoring all attention outputs amplifies the task readout to 0.7490. This hybrid keeps changed source embeddings while substituting receiver attention additions, altering the MLP trajectory and potentially removing cancellation. It is not evidence that attention is irrelevant or harmful. The experiment alone does not distinguish cancellation from other nonlinear background effects.

For key renames, all-attention and all-output restorations return local values exactly to receiver values. This is an expected graph-cut control: source embeddings are unchanged, attention imports context, and MLPs are token-local. Restoring each attention output at those positions makes subsequent MLP inputs match the receiver. The zero result verifies the implementation; it is not discovery of a minimal semantic circuit. Task edits change source embeddings and repeated x0 inputs, explaining why an all-output reset need not eliminate their effect.

Audits: 9,024 model forwards, 480 aligned edits (256 task, 96 nuisance, 128 identity), with the same 32 excluded nuisance alignments. Both expected nuisance resets, previous single-site replay, native logits/local values, exact output replacement, identity interventions, and the H780 target readout all match exactly. Metrics are finite and model/input/source hashes unchanged. The original reserved test was not used; natural-document pool remains 26,112. Groups were selected from earlier screens and this panel is exploratory, not independent confirmation.

Research implication: verified reusable units will need to preserve transformations across computational backgrounds, not merely reconstruct activation clouds. The current evidence favors an interacting, context-conditioned value pathway over independent additive module contributions; it does not yet verify a manifold, hierarchy, or computational DAG featurizer.

Next: inspect the exact bilinear MLP algebra at L1/L4 and partition observed changes into the two linear-in-change terms and their interaction, including RMS-normalization effects separately. Validate numerical reconstruction before causal readouts. Then freeze a specific mechanistic prediction and test fresh lexical/context combinations and additional tasks, rather than optimizing further on this copying panel. Return to featurizer/manifold comparisons using those transferable causal criteria.
