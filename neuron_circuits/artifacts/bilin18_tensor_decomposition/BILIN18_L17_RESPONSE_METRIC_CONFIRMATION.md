

## H425 — Larger fresh factorial confirmation

All-new1024 recipient documents exclude9920 earlier documents by independently rehashed full-document identity. Twelve new donor assignments and Gaussian draws, frozen H421 donor moments and H410 groups. Each donor assignment consists of two independent permutations of the512 H413 donors, one per512-recipient block. All four response/metric cells are newly computed. Same amplitudes0.25 and4, same joint3 contribution family and thresholds as H424. No interim stopping. Total completed documents10944. This is fresh recipient/draw confirmation, conditional on the same fixed donor pool and selected groups.

For each response full vector r and retained vector r_g, quadratic error is0.5 times the baseline output-Fisher energy of r-r_g, and quadratic effect is0.5 times that of r. Exact KL compares native output distributions after adding r or r_g to the baseline state. The linearized response uses double native-epsilon derivatives; exact output evaluation casts those response increments to native float32. Normalized recovery and S=R_union-R_core-R_added are computed within each cell, so denominators differ across cells. This is a controlled counterfactual accounting, not a unique mediation fraction.

Let I denote the H423 Gaussian-versus-raw amplitude interaction in S. With N=native response, L=linear response, K=exact KL, Q=quadratic metric, freeze three contributions: M=I_NQ-I_LQ, O=I_LK-I_LQ, X=I_NK-I_NQ-I_LK+I_LQ. Their sum is I_NK-I_LQ. A joint max-absolute95% band uses4000 paired document/draw bootstrap samples. Negative contribution requires joint upper<-0.02; equivalence requires its joint interval within±0.02. Linear/quadratic amplitude invariance and factorial closure require error<1e-10.

|Response|Metric|Distribution-by-amplitude interaction I|
|---|---|---:|
|Native|Exact KL|-0.056430|
|Native|Quadratic|-0.048076|
|Linear|Exact KL|-0.035624|
|Linear|Quadratic|0.000000|

|Contribution|Estimate|Joint95% interval|Negative beyond0.02|Equivalent within0.02|
|---|---:|---|---|---|
|MLP_only|-0.048076|[-0.089061, -0.007090]|False|False|
|output_only|-0.035624|[-0.076610, 0.005361]|False|False|
|factorial_interaction|0.027270|[-0.013715, 0.068255]|False|False|

Outcome: the MLP-only contribution is -0.048076, joint95% interval [-0.089061,-0.007090]. Its interval excludes zero in the negative direction. However, its frozen negative-beyond0.02 gate still fails: the upper bound is not below -0.02. Output-only is -0.035624, [-0.076610,0.005361], and factorial interaction is +0.027270, [-0.013715,0.068255]; both remain unresolved in sign. All three negative02 gates and all three equivalence02 gates remain failed. Reporting the MLP interval's sign does not replace or rename those gates.

The joint half-width decreases from0.063756 in H424 to0.040985 here, a35.7% reduction. Point estimates remain close to H424 (-0.042639,-0.039295,+0.031669). Thus broader recipient and source coverage gained precision rather than producing a qualitatively different decomposition. The native/exact-KL distribution-by-amplitude contrast is descriptively -0.056430; native/quadratic is -0.048076, linear/exact-KL is -0.035624, and linear/quadratic is exactly zero. All four cells were recomputed on the new panel.

The result supports a nonzero contribution from nonlinear MLP response generation within this diagnostic comparison. Exact output KL nonlinearity is not required for a negative trend to appear: the native-response/quadratic-score comparison has a negative joint interval. It does not establish the MLP as the only contributor, establish a minimum0.02 contribution, or make its effect a unique mediation fraction. Normalized scores use cell-specific denominators. The fixed donor calibration pool and selected head6 factor groups still limit scope.

For the broader question of computational units, this reinforces the distinction between algebraic factors and their functional roles. A factor product is a fixed component of the bilinear parameterization, but the predictive role of a group depends on the operating context, input normalization, intervention size and output metric. These experiments support studying context-conditioned response operators alongside native factors. They have not identified universal semantic units or shown that a particular partition is uniquely privileged.

Next discriminating experiment: keep the output Fisher metric fixed and cross native versus linearized changes in the pre-MLP normalized input with full bilinear response versus its first-order expansion. If x is the intact normalized input and u its change, the exact bilinear change is 2B(x,u)+B(u,u). This splits input-RMS nonlinearity from bilinear curvature while preserving the native local linear response as the shared baseline. The fully linearized cell should reproduce the existing amplitude-invariant baseline. Freeze the contrasts and numerical identity checks before new outcomes. This directly tests A1-A3 from H424 rather than running another undirected precision increase.

The earlier fixed-denominator polynomial design remains an optional control, but its first-order response differs from the native RMS derivative. The normalized-input factorial above avoids conflating that baseline change with curvature. No new experiment has yet been launched for this split.

Hypothesis priorities: A1-A3 (input normalization and bilinear response generation) move ahead for the next mechanistic experiment. A4-A6 (output RMS, cap and softmax scoring) remain plausible and unresolved, not refuted. A11-A16 (sampling moments, donor calibration and recipient-conditioned geometry) remain scope/uncertainty alternatives. A20 (higher-order semantic organization) is unsupported. The experiment's increased quantity improved a specific inferential distinction; it did not turn the factor groups into a universal ontology.


Audits: {'linear_quadratic_invariance': 0.0, 'factorial_closure': 0.0}. Independent arrays: {'all_four_cells_newly_computed': True, 'document_disjointness_verified': True, 'donor_block_permutations_verified': True, 'factorial_scores_and_joint_band_replayed': True, 'linear_quadratic_row_scaling_max_abs': 0.0}. Completed unique documents now10944. The broad research goal remains open.
