

## H342 — Context interaction versus edit curvature

Reused H339 head6 recipients and six realizations each of moment and donor controls; reused the H341 intact-state output metric. The four output modes [0, 2, 5, 1] were selected on this same panel in H341. This is exploratory decomposition, not independent confirmation.

For the quadratic compiler f, normalized intact input x, normalized edited input x+d, and frozen discovery mean μ, decompose the exact response as Df(μ)d + 2B(x−μ,d) + B(d,d). These are mean-linear, context-linear, and edit-quadratic terms. Their sum replays the compiler coefficient difference. The mean/context split depends on μ; edit-quadratic describes curvature in normalized input coordinates, not curvature along the raw residual intervention.

|Modes|Control|Mean-linear excess allocation|Context-linear excess allocation|Edit-quadratic excess allocation|Fidelity without edit-quadratic|
|---|---|---:|---:|---:|---:|
|all|moment|9.27%|88.57%|2.16%|99.56%|
|all|donor|8.24%|89.17%|2.59%|99.53%|
|top4|moment|1.53%|95.01%|3.47%|99.41%|
|top4|donor|-1.34%|97.11%|4.22%|99.35%|

Signed allocations split interactions equally and divide by control-minus-actual total energy. They are not causal percentages. Fidelity is 1−energy(omitted quadratic term)/energy(full response), using the local intact-state Fisher metric.

The mean/context response cosine for all16 modes is −0.784 on actual payloads, versus −0.432 / −0.436 on moment/donor controls; for selected4 it is −0.837 versus −0.547 / −0.563. Thus both larger context-linear energy and weaker cancellation deserve follow-up. Removing edit curvature retains 97.25% of actual full16 response energy fidelity and >99.5% for both controls.

Locked descriptive gates: top4_context_excess80: PASS, top4_curvature_allocation_under10: PASS. These concern the same exploratory panel, with no new confidence intervals.

Audit: coefficient-term replay relative error 6.9e-06; inherited H339 compiled-energy replay error 1.41e-06. Per-site terms and per-document term Grams saved. No native weight changes.

Next discriminator: test the full local compiler Jacobian restricted to the head6 source span. Compare actual and matched control edit energy in its sensitivity directions, then split changes in context-response magnitude from mean/context alignment. A frozen fresh-document replication should precede interpreting selected modes as stable units. The broad research goal remains active.
