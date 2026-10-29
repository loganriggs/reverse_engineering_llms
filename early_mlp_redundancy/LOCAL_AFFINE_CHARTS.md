# Local affine charts: construction and natural-text results

M79 is completed: all three frozen models, natural-text evaluation, task interventions, and local routing decomposition are audited. The frozen chart comparison stops here; no cluster-count, seed, or routing sweep follows.

## Theory and question

For the native quadratic product write F(z)=D[(Lz)⊙(Rz)], the Taylor chart at anchor c is T_c(z)=F(c)+J(c)(z-c). Exactly, F(z)-T_c(z)=F(z-c). This relates local approximation error to residual input geometry. The bound ||F(z-c)|| ≤ ||D|| ||L|| ||R|| ||z-c||² makes covering the input distribution a motivated heuristic; it does not guarantee small actual error or downstream fidelity.

Nearest-anchor routing can itself be discontinuous. For F(x,y)=(x²,xy), charts centered at (-1,0) and (1,0) differ by2 across the routing boundary at y=1, even as the native output difference approaches zero. A toy and100 random identity tests establish this possibility, not its prevalence in the model.

## Frozen construction

Use exactly4,092 training positions from natural documents0–31. Compare the training mean,32 sampled training anchors (seed7901), and32 k-means centroids initialized with those same anchors, using50 Lloyd steps. Routing uses raw normalized-input Euclidean distance, with fixed tie and empty-cluster rules. All anchors and explicit matrices were frozen before evaluation. No task inputs or labels enter fitting.

Training mean squared nearest-anchor distances: mean148.949, random187.488, k-means108.400. Random data points are not automatically good covering centers in this input geometry. Independent direct-distance reconstruction reproduces all50 k-means steps and all4,092 assignments per model. Every one of65 stored charts passes factored-native derivative and exact-remainder checks; maximum relative discrepancy3.57e-15. Eight native preflight forwards verify float32 predictions (relative discrepancy below2.55e-8), exact callback/whole-MLP override agreement, native product bypass, and exact model restoration.

## Held-out natural text

Same32 held-out documents,128positions each. All96 candidate outputs are saved and independently audited, including direct-distance routing and native factored-polynomial predictions.

| Model | Output KL | CE increase | Top-1 agreement | Mean-document relative write error |
|---|---:|---:|---:|---:|
| Mean-anchor chart | 0.48455 | 0.51019 | 63.92% | 0.87667 |
| Random32 charts | 1.54929 | 1.58126 | 40.14% | 1.20754 |
| K-means32 charts | 0.15157 | 0.15693 | 79.81% | 0.48266 |
| M78 fitted affine reference | 0.16322 | 0.16844 | 79.57% | 0.58032 |

K-means helps substantially relative to these chart controls and modestly relative to the fitted affine reference. This is descriptive evidence on one frozen cohort; no statistical population-level superiority claim is made. In particular, clustering does not yet establish semantic manifolds, shared computations, neuron redundancy, or cancellation.

The32-chart models store42,504,192 slope/intercept coefficients each, plus anchors. They reuse native weights to construct their operators and exceed native MLP parameter count. This is a geometric reference experiment, not a compression or matched-capacity victory. The1152 active-coordinate field describes the input dimension, not1152 semantic features.

## Completed task and routing evaluation

All56 pairs/seven training-excluded families/14formats were retained, including weak competence cases and24 source-ineligible model assignments. There were2,608 task forwards,112 exact native-baseline replays,328 exact teacher-target replays,2,952 audited response vectors,984 geometry rows, and336 independent native-polynomial prediction checks. Maximum applied prediction relative discrepancy is2.61e-8; maximum metric discrepancy is7.45e-9. Including prior natural and preflight work, M79 uses2,744 native forwards.

Normalized causal-response error compares changes in final log probabilities, weighted by native output probabilities. Error1 is the zero-response baseline, not sufficient fidelity. Source patches changed input positions; final patches the last position. A “both replaced” response uses reconstructed backgrounds and reconstructed changes, centered on the candidate's own background. The table uses equal-family averages.

| Model | Native background / predicted change: source | Same: final | Both replaced: source | Both replaced: final |
|---|---:|---:|---:|---:|
| Mean chart | 0.7930 | 0.9896 | 3.9304 | 2.4763 |
| Random32 | 0.8772 | 1.2706 | 7.0583 | 22.5434 |
| K-means32 | 1.2400 | 0.6486 | 42.3270 | 1.4540 |
| M78 affine reference | 0.6932 | 1.9224 | 13.4096 | 2.4733 |

For the five contrast-qualified families, k-means scores0.6973/0.5504 for predicted changes on native backgrounds, but58.5886/1.5135 when both are replaced (source/final). Pooled errors for both replaced are7.9463/0.6116 on qualified families and7.4105/0.6139 on all seven. Retain both averages: weekday-successor source responses account for much of the equal-family failure (288.38); it is not solely a weak-task artifact. K-means qualified contrast accuracy is91.25% versus native96.25%, with22.5% versus41.25% top-1 answer accuracy. Task intact KL0.3097 is worse than the affine reference0.2133, despite the modest natural-text advantage. No uniform superiority claim is warranted.

### What routing explains—and does not

Exactly decompose the predicted donor-minus-recipient write into a change within the recipient's chart plus the change from selecting the donor's chart. An independent factored-native-polynomial calculation verifies all648 nonidentity decompositions (maximum relative metric discrepancy5.82e-15).

- K-means switches on24/138 source positions (17.39%), including14/20 weekday positions. Source pooled local change error is0.6796 for the within-recipient term and0.7205 after adding routing. This is a local-vector comparison, not a downstream causal ablation.
- Random anchors switch on26/138 source positions; the analogous errors are0.8931 and1.0767.
- Neither32-chart model switches on any of112 final-position assignments. Their remaining final-position causal errors therefore cannot be attributed to donor/recipient chart crossing. This does not rule out background errors arising from routing at other sequence positions.
- On weekday source positions, k-means within-chart error0.8099 rises to1.1663 with routing. The large downstream failure is compatible with several explanations, including background incompatibility and nonlinear amplification; routing alone is not established as its cause.

Task input distances are higher than natural held-out distances for all three models. K-means mean squared distance is140.063 on full task endpoints versus113.121 on natural text. These descriptive distances support measuring shift but do not isolate its effect.

## Conceptual conclusion and next direction

Input geometry provides useful approximation structure, but better coverage and natural reconstruction do not establish computational equivalence. Routing is one failure mechanism; substantial error without final-position switches shows it is not the whole account. These experiments concern function geometry and downstream use, not literal neuron-removal redundancy or native signed cancellation.

Stop this chart comparison. The next deliverable is a predeclared executable-input featurizer comparison with diverse competent training tasks and joint background/change causal supervision, retaining all seven whole-task exclusions. First audit the available training targets, information access, budget controls, and output-oracle ceiling. No new architecture or training run is frozen yet. [Twenty conceptual hypotheses](CONCEPTUAL_HYPOTHESES_M79.md) separate supported findings, live explanations, and discriminating tests.

Artifacts: validation/local_affine_chart_protocol.json, local_affine_chart_fit.json, local_affine_chart_fit_audit.json, local_affine_chart_forward_preflight.json, local_affine_chart_natural.json, and local_affine_chart_natural_audit.json. Large states and all outputs remain in the research artifact directory.

Additional artifacts: validation/local_affine_chart_factorial.json, local_affine_chart_factorial_audit.json, local_affine_chart_comparison.json, local_affine_chart_routing.json, and local_affine_chart_routing_audit.json.
