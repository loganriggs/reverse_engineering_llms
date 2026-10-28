

## H424 — Separating nonlinear response and output scoring

Reuse H423 recipients, donor mappings, Gaussian draws and H410 groups. Cross native finite MLP responses with their local linearization, and exact native output KL with a fixed baseline Fisher quadratic metric. Amplitudes remain0.25 and4. Direct residual stays fixed. The native/exact-KL arrays are reused from H423 with a frozen file hash; they were not independently rerun. All other cells are newly computed. No fresh documents or factor selection.

For each response full vector r and retained vector r_g, quadratic error is0.5 times the baseline output-Fisher energy of r-r_g, and quadratic effect is0.5 times that of r. Exact KL compares native output distributions after adding r or r_g to the baseline state. The linearized response uses double native-epsilon derivatives; exact output evaluation casts those response increments to native float32. Normalized recovery and S=R_union-R_core-R_added are computed within each cell, so denominators differ across cells. This is a controlled counterfactual accounting, not a unique mediation fraction.

Let I denote the H423 Gaussian-versus-raw amplitude interaction in S. With N=native response, L=linear response, K=exact KL, Q=quadratic metric, freeze three contributions: M=I_NQ-I_LQ, O=I_LK-I_LQ, X=I_NK-I_NQ-I_LK+I_LQ. Their sum is I_NK-I_LQ. A joint max-absolute95% band uses4000 paired document/draw bootstrap samples. Negative contribution requires joint upper<-0.02; equivalence requires its joint interval within±0.02. Linear/quadratic amplitude invariance and factorial closure require error<1e-10.

|Response|Metric|Distribution-by-amplitude interaction I|
|---|---|---:|
|Native|Exact KL|-0.050264|
|Native|Quadratic|-0.042639|
|Linear|Exact KL|-0.039295|
|Linear|Quadratic|0.000000|

|Contribution|Estimate|Joint95% interval|Negative beyond0.02|Equivalent within0.02|
|---|---:|---|---|---|
|MLP_only|-0.042639|[-0.106395, 0.021118]|False|False|
|output_only|-0.039295|[-0.103051, 0.024462]|False|False|
|factorial_interaction|0.031669|[-0.032087, 0.095426]|False|False|

Outcome: all three negative-beyond0.02 gates and all three equivalence-within0.02 gates fail. No individual contribution is established under the frozen joint band. Point contributions are MLP-only -0.042639, output-only -0.039295 and factorial interaction +0.031669, summing to the H423 native effect -0.050264. Joint95% intervals are [-0.106395,0.021118], [-0.103051,0.024462] and [-0.032087,0.095426]. Linear/quadratic amplitude invariance and factorial closure are exactly zero; saved quadratic rows scale exactly by256 between amplitudes0.25 and4. The native/KL cell reuses H423 and is not a new replication of its sign.

This rules out an implementation story in which the deliberately linear/quadratic counterfactual develops an amplitude effect. It does NOT statistically rule out either nonlinear component being negligible, nor does it establish that both are necessary. At the point-estimate level, either response generation or output scoring can create a negative trend alone, with a positive factorial term offsetting their sum. Normalized scores use each cell's own effect denominator, so those three numbers must not be called unique causal mediation contributions or proportions of a semantic mechanism.

A post-result precision diagnosis resampled documents alone or donor/Gaussian draws alone, retaining the original frozen joint analysis. Document-only bootstrap standard deviations for [MLP,output,interaction] are [0.01056,0.01506,0.01748]; draw-only values are [0.00817,0.01664,0.01409]; joint values are [0.01714,0.02828,0.02784]. Output and factorial-interaction statistics set the joint maximum in1708 and1663 of4000 replicates, versus629 for MLP. Both recipient variation and source-draw variation matter. These resampling comparisons are a precision diagnostic, not an additive variance decomposition or replacement significance tests.

The leading mechanistic possibilities remain input normalization/bilinear response generation (A1-A3) and output normalization/cap/softmax scoring (A4-A6), potentially combined. Finite-sample moments and recipient-conditioned geometry (A11,A12,A16) remain live explanations of some distribution dependence. Semantic organization (A20) remains unsupported. The experiment narrows the useful next action: more precise measurement of these two broad mechanisms before subdividing output scoring into several noisy tests.

Next: predeclare a balanced precision follow-up that increases BOTH recipient coverage and independently generated source draws, retaining this same three-contrast family and thresholds. Reusing H423 recipients as part of a larger pool makes that follow-up cumulative precision work, not wholly fresh confirmation; preserve that distinction. A separate all-new confirmation would be needed for a selected mechanistic attribution. After precision improves, split final RMS, logit cap and softmax only if output nonlinearity remains supported; split input RMS and bilinear curvature if response generation remains supported. Exact empirical moment matching is a useful orthogonal control but does not remove recipient-assignment uncertainty.
### Twenty mechanistic alternatives for the amplitude effect

These alternatives can coexist. They are hypotheses, not posterior probabilities or established mechanisms. A1-A20 name this mechanistic shortlist rather than replacing the global twenty-hypothesis tournament.

|ID|Possible explanation|Experiment that would distinguish it|
|---|---|---|
|A1|Input RMS changes radial gain as edits grow.|Compare exact RMS with a frozen denominator in otherwise identical finite MLP responses.|
|A2|The source-residual angle controls the input RMS denominator.|Hold source norm fixed and vary its residual projection with paired directional controls.|
|A3|The bilinear quadratic source term changes the group relationship.|With input normalization fixed, compare full bilinear response with its first-order term.|
|A4|Final output RMS causes the distribution-dependent trend.|Linearize only final RMS while retaining logit capping and exact KL.|
|A5|The tanh logit cap changes which errors matter.|Linearize only the cap around baseline logits, retaining output RMS and exact distribution scoring.|
|A6|Softmax/KL curvature causes the trend even for linear MLP responses.|Compare exact KL and baseline Fisher scoring of the same linear response; then isolate normalization and cap.|
|A7|A few baseline logit margins determine sensitivity to nonlinear scoring.|Stratify by frozen margin bins and validate on a separate recipient panel.|
|A8|Natural and Gaussian source norm tails differ.|Use controls with matched radial distributions and separately audited directional moments.|
|A9|Directional fourth moments matter even at matched norms.|Compare covariance-matched spherical-direction controls with natural directions within narrow norm bins.|
|A10|Source coefficient skewness creates asymmetric finite responses.|Pair positive and negative source edits; odd-order effects should reverse differently from even-order effects.|
|A11|Finite sampled Gaussian covariance mismatch is sufficient.|Use exactly whitened/recolored Gaussian-derived batches and paired assignment controls.|
|A12|Finite sampled Gaussian mean mismatch is sufficient.|Center each sampled batch to the donor mean while retaining its original covariance, then compare to uncentered draws.|
|A13|Token-position mixtures conceal opposing mechanisms.|Estimate the predeclared contrast separately by position with simultaneous uncertainty and independent replication.|
|A14|The fixed donor calibration pool is atypical.|Repeat calibration using a new disjoint donor pool without changing selected groups or contrast.|
|A15|A small set of recipient documents dominates the effect.|Measure influence and report robust or trimmed sensitivity separately without replacing the original statistic.|
|A16|Recipient-conditioned response operators interact with donor assignment.|Average controlled source ensembles within fixed recipients; compare with pooled assignment estimates.|
|A17|The added factors compensate for core errors only along some directions.|Hold the core fixed and evaluate signed cross-response structure across independently selected directional strata.|
|A18|The effect is specific to the selected core/add partition.|Repeat the frozen contrast for independently chosen matched-size partitions; distinguish selection effects from universal grouping.|
|A19|The effect is specific to head6 or layer17.|Use a new selection panel for another source/layer, followed by separate confirmation.|
|A20|Higher-order semantic organization contributes beyond these geometric mechanisms.|After moment and nonlinear-scoring controls, use controlled linguistic edits with predicted, held-out behavior; current Gaussian comparisons alone cannot establish this.|

Floating-point artifacts remain an implementation alternative rather than a proposed model mechanism; exact scaling, row replay, native-epsilon derivatives and partition audits constrain that possibility but do not substitute for scientific controls.


Audits: {'linear_quadratic_invariance': 0.0, 'factorial_closure': 0.0}. Independent arrays: {'native_KL_reference_rows_identical': True, 'H423_primary_replayed': True, 'factorial_scores_and_joint_band_replayed': True, 'linear_quadratic_row_scaling_max_abs': 0.0}. Completed unique documents remain9920. The broad research goal remains open.
