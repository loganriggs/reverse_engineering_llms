

## H268 — Exact accounting locates missing response outside the selected core

The next information-gaining test used no fitted readout. Center the native-minus-compiled residual polynomial E at the frozen discovery mean. Split normalized states into the H267 core16 projection, tensor-selected partner24 projection, and remainder. Decompose E(z)−E(x) into six terms: retained624 mean-linear response, omitted mean-linear response, within-core quadratic change, core–selected-partner change, core–omitted-partner change, and entirely outside-core quadratic change. Record their full signed Gram matrix, not just individual energies.

Same128 H259 documents, four sites, nine head sources, actual and same-head norm/angle controls, edit scales0.1/1.0. This is an adaptive analysis on reused documents, not fresh replication. Float64 polynomial arithmetic starts from float32 model weights and normalized states. Exact weight access makes this a diagnostic, not a new cheap predictor or behavioral replacement.

| Structural truncation | Small actual edit | Full actual edit |
|---|---:|---:|
|model_shape|21.71%|22.34%|
|model_shape_full_linear|23.08%|23.79%|
|core_all_partners|33.37%|34.29%|
|outside_plus_linear|71.31%|70.09%|
|all_quadratic_linear624|96.89%|96.92%|
|full|100.00%|100.00%|

Locked contrasts (percentage-point gain, paired document-bootstrap95%):

- omitted_partner_gain: actual_s0.1: 10.30 [9.68, 10.91]; actual_s1.0: 10.49 [9.89, 11.09]; gate FAIL.
- outside_core_needed: actual_s0.1: 66.63 [63.67, 69.33]; actual_s1.0: 65.71 [62.72, 68.50]; gate PASS.
- linear_restriction: actual_s0.1: 1.37 [1.14, 1.60]; actual_s1.0: 1.45 [1.21, 1.68]; gate FAIL.

Only the outside-core gate passes. Restoring every core partner helps about10 points, below its20-point gate; relaxing the linear restriction helps about1–2 points, below its10-point gate. Restoring outside-core quadratic terms adds about66 points. Thus the current core misses substantial exercised computation even if supplied with every partner. This is a statement about this fixed centered decomposition and population, not a basis-independent intrinsic dimension.

Individual term energies sum to about1.12 times actual response energy; cross terms supply modest net cancellation. The outside-core term has about66% of response energy and about63% signed alignment with the target. Cancellation is present but does not hide an order-of-magnitude sum of large terms here. Angle controls show the same qualitative outside-core dominance. The full signed matrices and document-level arrays remain available for alternative weighting analyses.

Maximum six-term relative replay error is9.0e-12. Four relevant algebra/solver tests pass (not the full project suite). The accounting test includes a nonzero mean and arbitrary compiler offsets, linear terms and bias. Native weights and original compiler remain unchanged. All runs complete.

**Interpretation and limits.** The leading explanation is missing interactions beyond the selected core, with coefficient estimation also supported by prior data-scaling results. Missing partners alone and the624-linear restriction are weaker explanations for this structural truncation. Crucially these truncations are NOT best-fit bounds: a learned readout can absorb omitted-term correlations, so H267's42% learned fidelity can exceed the exact-coefficient model-shape fidelity without contradiction. Neither response energy nor this algebra establishes semantic features or downstream importance.

**Next experiment: discriminate a poor core from broadly distributed residual interactions.** Freeze nested discovery-only residual-Jacobian cores of widths8/16/32/64/128 and matched natural-PC cores. For each, calculate exact full mean-linear plus every quadratic term touching the core, and its outside-core complement on identical source responses. The primary curve is pooled response fidelity, with document-bootstrap intervals and signed cancellation. If either family reaches80% at width<=32, a compact but previously misselected core remains plausible; if both stay below50% through width64, these two candidate core families remain inadequate. This would not rule out all alternative low-dimensional representations. Compare small/full actual edits and matched-angle controls; use a fresh, document-disjoint panel to confirm any selected width before claiming replication. No fitted-coefficient or semantic claim follows from the oracle sweep. Only after locating the necessary interactions should a matched-budget predictor be trained.

The twenty-hypothesis map is maintained rather than replacing it with twenty new stories after each result. H268 narrows the current branch of that map: outside-core structure now outranks omitted partners and linear restriction; sampling, metric dependence, semantics and external validity remain separate open questions. Broad research goal remains active.
