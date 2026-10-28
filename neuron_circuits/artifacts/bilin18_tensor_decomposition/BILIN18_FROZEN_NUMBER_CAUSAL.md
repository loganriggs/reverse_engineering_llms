## H888 — Frozen number coordinates causally support native agreement on new nouns

Directly changing the native MLP number coordinates in the prespecified L11–16 set reduces the agreement margin by about half on irregular and new regular nouns. Correct-number clamps and random-direction controls preserve the margin much more closely, while flipping the complementary layers has small or inconsistent effects. The model still prefers the number-correct candidate on every tested endpoint. This establishes a transferable causal contributor, not a sufficient or uniquely necessary agreement circuit.

### Frozen transfer design

Use only H887's originally eligible task cells: irregular B/C and new-regular A/B, totaling60 endpoints. Both original trained axes A/B are applied in every cell, including C; neither is selected using these outcomes. Frozen means, first PCA directions and singular/plural coordinate centroids come from the four original discovery noun pairs. Core layers L11–16 and complement L8–10/L17 were declared before the run.

Two distinct interventions are evaluated:

- **Direct native computation:** at the final token, replace the coordinate of the current normalized MLP input with the correct number-centroid, the opposite centroid, or their midpoint. Preserve its orthogonal complement locally. Apply the exact change in the bilinear MLP output, including the quadratic term. There is no cached-native-output clamp and no L8H3 headwrite in these conditions: they modify the unperturbed model's computation directly.
- **Response computation:** alter the native partner coordinate inside the local MLP Jacobian while preserving its native output offset, then apply the same symmetric L8H3 headwrite as earlier. Compare with selected-layer self-linearization. These conditions test response changes, not unperturbed agreement necessity.

A direct identity control makes no coordinate change. A random control uses an orthogonal random direction per axis/layer and the same local displacement magnitude that the flip rule would prescribe at that current state. Earlier intervention effects can change subsequent states, so this is local magnitude matching, not identical total trajectory length. One random draw per axis/layer is a limited control, not a distribution over all random controls.

### Direct native effects in L11–16

Margins are expected-number logit minus opposite-number logit, averaged across both selected templates and endpoints within cohort. Positive margins favor the correct number.

|Cohort|Frozen axis|Native margin|Correct centroid|Opposite centroid|Midpoint erase|Random direction|
|---|---|---:|---:|---:|---:|---:|
|irregular|A|4.783|4.765|2.398|3.333|4.915|
|irregular|B|4.783|4.488|2.555|3.620|4.608|
|new_regular|A|5.339|5.325|2.569|3.854|5.214|
|new_regular|B|5.339|4.995|2.591|4.381|5.190|

Opposite-centroid clamps reduce irregular margins from4.783 to2.398/2.555 and regular margins from5.339 to2.569/2.591. Correct-centroid margins remain4.488–4.765 and4.995–5.325, respectively. Random controls also remain close to native. Erasure gives intermediate effects.

Flipping the complement yields irregular margins5.324/4.662 and regular5.646/5.127, much smaller or opposite changes. This supports the preselected later layer set, but compares a six-layer set with a four-layer set and does not identify individual-layer necessity or additivity.

Binary is/are accuracy remains100% in every direct condition. Full-vocabulary exact is/are top-1 drops from75% native to46.4%/64.3% for irregular nouns and59.4%/62.5% for regular nouns under core flips. Thus reducing number preference can allow other vocabulary items to win without selecting the opposite-number verb. No claim of systematic verb-number reversal is supported.

Native-to-edited vocabulary KL under core flips is approximately0.088–0.125, versus0.007–0.018 for correct-centroid clamps and0.021–0.034 for random controls. The direct changes are meaningful distributional interventions, not merely a readout rescaling.

### Response intervention corroboration

Within the core set, correct-centroid response errors relative to self-linearized signed responses are32.4–32.5% for irregular and14.8–24.8% for new regular nouns. Opposite-centroid errors rise to168.8–175.0% and131.3–141.1%. In the complement, opposite-centroid errors are smaller, roughly29–54%, and sometimes little different from correct-centroid replacement. These are perturbation-response comparisons and must not be conflated with the direct native-margin effects above.

### Interpretation and remaining work

A direction first found by unsupervised variance in context-specific partner activations has a discovery-only number interpretation, decodes irregular and regular transfer nouns, and causally contributes to native agreement when changed in a prespecified layer set. This is a more substantive candidate computational component than reconstruction quality or label correlation alone.

Important limits remain: there is one direction per layer and trained template, not one universal feature; correct clamping does not perfectly preserve native computation; all native binary decisions remain correct; and joint interventions do not establish unique mediation. The model has other number-supporting computation or enough residual margin to remain correct. These results do not yet validate a block-sparse ontology or an automatically discoverable hierarchy/DAG of units.

Next quantify dose response and individual-layer contributions with fixed intervention strengths and stronger random/norm controls, and test selectivity against copying the same nouns and non-number output contrasts. Separate changes caused by the bilinear coordinate from changes in normalized-input norm. The outstanding copying lexical-coverage learning curve remains necessary to understand where the compact partner approximation fails.

### Verification

Completed 3,420 full-model forwards. Native and identity logits replay exactly. Coordinate clamp error is at most 7.46e-14; local magnitude matching error 3.73e-14; random-axis orthogonality error 2.64e-16. Independent direct-update audit on all60 native states,10layers andbothaxes compares the expansion with explicit bilinear evaluation: double-algebra relative error 3.71e-15; actual float-module relative error 1.38e-06, with max coordinate discrepancy 0.0293 retained for the large update vectors. Hashes unchanged; all selected outputs/KL finite; minimum headwrite amplitude 0.01799, none dropped. No reserved test evaluated; natural-document pool unchanged at26,112. Broad research goal active.
