## H889 — Number-axis effect survives norm controls, is selective for agreement, and concentrates at L12

The direct agreement effect from H888 is almost unchanged when total normalized-input norm is preserved exactly. It increases across the fixed quarter/half/full dose ladder, exceeds four locally matched random controls, and is much larger for agreement than for copying the same nouns. Layer12 is the largest single-layer contributor in every cohort/axis comparison. These results strengthen the candidate component while still falling short of a complete or uniquely necessary agreement circuit.

### Design and controls

Use all120 task/copy endpoints in the originally eligible H887 cells: irregular B/C and new-regular A/B. Apply both frozen task-trained axes to every context, including copying; no copying-specific direction is fitted. At each selected MLP final-token input, interpolate the coordinate toward the frozen opposite-number centroid by doses0.25,0.5,1.0.

Compare the raw coordinate change with an edit that simultaneously reaches the same target coordinate and preserves the original total input norm. The latter rescales the coordinate-orthogonal component to stay on the same sphere; it therefore is not a pure one-direction displacement. A radial-only control changes only overall norm to match the raw edit's norm. Four random controls per axis/layer move along random tangent directions with the same norm and chord distance as the number-directed spherical edit at that current state. Later trajectories need not match, because earlier edits alter subsequent inputs.

CoreL11–16 receives the dose/control panel. Each of its six layers and the L8–10/L17 complement also receives full-dose raw and sphere edits. All conditions were declared before this run's outcomes. Random controls use four draws, not an exhaustive null distribution. No refitting, task-label-based pair filtering, or axis selection occurs.

### Full-dose agreement and copying effects

Values are reductions in the appropriate native expected-answer logit margin. Agreement uses is versus are; copying uses the original source noun versus its singular/plural counterpart.

|Cohort|Axis|Agreement raw drop|Agreement norm-preserved drop|Norm-only drop|Copy norm-preserved drop|Random norm-preserved drop range|
|---|---|---:|---:|---:|---:|---|
|irregular|A|2.385|2.391|-0.010|0.184|0.062–0.324|
|irregular|B|2.228|2.231|0.000|0.349|0.221–0.409|
|new_regular|A|2.771|2.774|-0.007|0.273|0.075–0.424|
|new_regular|B|2.749|2.751|-0.004|0.395|0.217–0.437|

Native agreement margins are4.783 irregular and5.339 regular. Full norm-preserved flips lower them by2.231–2.774, whereas matched copying margins fall by0.184–0.395 from native3.893/4.319. Copying effects are nonzero, so selectivity is relative rather than perfect. Agreement vocabulary KL is0.088–0.126 versus copying0.019–0.026 under the sphere edit. The norm-only control has negligible margin effect; norm inflation is not a sufficient explanation of the observed agreement change.

Across all cohort/axis comparisons, mean agreement margin drops increase monotonically with dose. At quarter dose they are0.461–0.777, at half dose1.361–1.796, and at full dose2.231–2.774. This is a mean dose response over fixed examples, not a guarantee of monotonicity for every individual example. Binary expected-answer accuracy remains100% in the reported full-dose conditions; the margin effect is substantial but does not establish decisive control of the final prediction.

### Individual layers, norm-preserved full dose

|Layer|Irregular axis A|Irregular axis B|Regular axis A|Regular axis B|
|---|---:|---:|---:|---:|
|11|0.196|0.154|0.512|0.422|
|12|2.038|1.400|1.882|1.126|
|13|0.391|0.401|0.329|0.349|
|14|0.197|0.192|0.170|0.165|
|15|0.044|0.133|-0.112|0.038|
|16|0.081|0.138|0.046|0.157|

L12 has the largest marginal effect in each comparison, with drops1.126–2.038. These single-layer effects are not an additive partition of the joint effect: downstream responses and interactions differ under each intervention. L12 is selected for follow-up from this result, so a later test must distinguish exploratory localization from prospective validation. The two axes also differ materially in L12 effect size despite both decoding number well, another reason not to equate classification accuracy with a unique causal direction.

### Interpretation and next computation test

The evidence now combines unsupervised discovery, frozen semantic labeling, irregular-noun transfer, direct causal effects, norm controls, dose dependence and relative task selectivity. It supports investigating the number direction as part of a conditional computation. It still does not establish a universal manifold, a complete agreement circuit, or a stable ontology across training seeds/models.

Next analyze L12's bilinear computation along the frozen direction. Writing normalized input as orthogonal context plus a scalar coordinate gives an exact decomposition into context-only output, a coordinate-linear cross term, and a coordinate-squared term. Audit that decomposition against the actual MLP output and use separate causal interventions to test which term carries the native margin effect. This can distinguish a reusable number-dependent operation from an axis that merely perturbs many unrelated computations. Keep both frozen axes and transfer nouns; avoid picking channels based on the test outcome without a new validation split.

The copying lexical-coverage learning curve remains outstanding. Its prior fidelity failure is not explained by the current agreement-selectivity experiment and remains part of the broader representation question.

### Verification

6,360 full-model forwards, matching the declared count. Native and identity logits replay exactly. Norm preservation error at most 2.13e-14; sphere coordinate-target error 7.28e-14; random displacement matching error 6.06e-13. No infeasible sphere targets. H888 raw-flip mean margins reproduce within 0. Exact bilinear output expansion is the independently checked H888 implementation. All hashes unchanged, outputs/KL finite; no reserved test evaluated; natural-document pool unchanged at26,112. Broad research goal active.
