# Local affine charts: construction and natural-text results

M79 is partially completed. All three frozen models are fitted and verified; the full natural-text evaluation is audited. Task interventions, chart-switch measurements, and the final scientific comparison remain pending. No cluster count, seed, or routing sweep is authorized by this protocol.

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

## Remaining discriminating work

Evaluate all56 pairs/seven training-excluded families/14formats at source and final positions under all three background/change combinations, retaining weak cases and ineligible alignments. Independently audit outputs. Record distances, occupancy, chart switches, and the exact decomposition into recipient-chart change plus routing change. A large routing term does not by itself establish harm: amplitude and direction confounds remain. Compare all three frozen models with M78 and fixed M76/M77 references. Then stop this chart comparison and decide whether to learn a constrained representation or return to targeted reusable-subcomputation tests.

Artifacts: validation/local_affine_chart_protocol.json, local_affine_chart_fit.json, local_affine_chart_fit_audit.json, local_affine_chart_forward_preflight.json, local_affine_chart_natural.json, and local_affine_chart_natural_audit.json. Large states and all outputs remain in the research artifact directory.
