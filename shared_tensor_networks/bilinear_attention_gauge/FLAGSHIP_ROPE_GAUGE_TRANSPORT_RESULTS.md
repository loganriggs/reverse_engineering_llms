# RoPE transport limits a naive component grouping on the 546M head

The [trained three-rule toy](THREE_COUNT_CAUSAL_GROUPING_RESULTS.md) recovered three unlabeled variable groups by grouping sparse axes according to their ordinary attention response. This follow-up applies that rule to layer 0, head 0 of the 546M bilinear-attention checkpoint. The immediate result is a **position-gauge problem**: individual components of a post-RoPE rank-96 QK1 cut change greatly when both token positions are translated, although the summed head response changes much less. An exact co-rotation transports the decomposition across the translation, but it also transports the cut. The experiment therefore does not establish semantic or physical component groups in the deployed fixed cut.

## Object and panel

Use the two existing full-width rank-96 QK1 cuts: one fitted under uniform token weighting and one under FineWeb token weighting. Each is SVD-factorized, then represented in the SVD gauge, three random orthogonal gauges, and six existing activation-sparse gauges (three identity and three random starts). All gauges within one cut sum to the **same rank-96 QK1 operation**. The response of component \(i\) is the complete one-head output contributed by its QK1 factor pair, including the unchanged QK2 factor and value/OV map. The latter is expressed in an exact 128-dimensional output-norm coordinate system; its squared norm matches the 768-dimensional OV output to at most \(5.9\times 10^{-10}\) on the checked values.

For each component, flatten the mean key-token response over query tokens and output coordinates. Absolute cosine between these vectors defines an unlabeled grouping graph. Fit at key/query positions (7, 8), then evaluate on disjoint query IDs and key samples at (7, 8), translated same-gap (31, 32), and changed-gap (32, 64). A fourth test co-rotates the factors by +24 positions before testing at (31, 32); this represents the **transported cut** \(R_{24}XR_{24}^{T}\), not the originally deployed fixed \(X\). Both uniform and corpus key laws, and their concatenated signatures, are evaluated.

The main panel uses 128 uniform plus 128 corpus query anchors in each split, and 8,192 uniform or corpus key draws per split. Query IDs are disjoint. Uniform key draws are disjoint; corpus draws come from disjoint FineWeb document ranges (fit 0:15,000, test 17,500:20,000), though frequent token IDs can occur in both. These are substantial *sampled* panels, not full-vocabulary or all-context enumeration. The earlier 16-query/512-key pilot is retained as a separate exploratory artifact.

## Transport diagnostic

To isolate position from token sampling, compare **identical** fit token IDs at (7, 8) and (31, 32). Relative squared error is normalized by the fit component-response energy (or summed-response energy for the total). Mean over both cuts and all ten gauges:

| Key law | Fixed components | Co-transported components | Fixed summed response | Co-transported summed response |
|---|---:|---:|---:|---:|
| Uniform | 1.09 | 0.0000081 | 0.00611 | 0.0000040 |
| Corpus | 1.25 | 0.0000098 | 0.00861 | 0.0000046 |

The near-exact co-transport is predicted by rotary covariance for a global translation with the same relative gap. The small remaining error is consistent with the checkpoint's rounded rotary tables. It is an algebraic control, **not independent evidence that the groups learned semantics**. The fixed component decomposition is unstable under this translation even when its sum remains fairly close. The fixed summed response is not exactly translation invariant because a fixed post-RoPE cut need not commute with the rotary shift.

In exact arithmetic, a post-RoPE score with a fixed bond matrix \(X\) has the form \(qR_p X R_t^\top k^\top\) (row-vector convention). Translating both positions by \(\Delta\) replaces \(X\) by its rotary conjugate in the original coordinates. A fixed cut is invariant for every query/key activation only if it commutes with that rotation. The same condition applies more stringently to each component matrix. A rank-one real component generally cannot be invariant under the nontrivial two-dimensional rotary frequency planes; a position-invariant scalar-axis ontology is therefore a restrictive assumption even when the sum of many axes is approximately invariant.

A separate **weight-only** check measures \(\|R^\top X R-X\|_F/\|X\|_F\) without selecting token IDs. At shift 24 this is **.370** for the uniform rank-96 cut and **.386** for the corpus cut. For the illustrative first activation-sparse axis it is **1.125** and **.990** respectively. These are not errors in the natural-language task; they quantify how far the fixed weight objects depart from exact rotary covariance. The full shift-1, shift-8, shift-24 control and a random-axis comparison are saved in `full_width_results/head0_rope_covariance_diagnostic.json`.

## Held-out graph transfer

For each cut and key mode, set a graph threshold to the 99.9th percentile of fit-panel cosine similarities from the three random gauges. At this threshold, the graph has only about 2–5 edges per gauge on average. Across the six cut/mode combinations, the sparse gauges have 2.9–3.6 edges on average, versus 4.7 for random gauges; their graph size is not a compelling multi-group partition. Conditional on a graph having at least one edge, mean edge retention on disjoint tokens at the same positions is 0.68–0.79 for the sparse gauges in each mode, versus 0.23–0.53 for the random gauges. That is a modest stability signal for a **small graph**, not a semantics result. The SVD gauge and random gauges also sometimes retain edges.

At translated positions with the **fixed** cut, mean sparse-gauge edge retention falls to 0–0.17 depending on key mode. Co-transport raises retention to exactly the same values as the same-position test, as expected from the rotary identity; it cannot validate the original fixed cut on new positions. At the changed relative gap, retention is 0–0.025. The fixed threshold sweep from .6 through .9 is in the JSON; the null-calibrated summary avoids selecting a favorable threshold after seeing the test.

The toy method therefore has **not transferred as a demonstrated interpretability method**. We have neither a semantic label test nor a selective full-model group ablation for these new graphs. It would be misleading to call the tiny graph a recovered task ontology.

## Next decision

Define a position-covariant explanatory object before another group search: either move the QK1 cut to a pre-RoPE location, describe components as rotary-transported families, or group a relative-offset-indexed response kernel. These are different choices. Keep a separate fixed-cut fidelity gate for the *actual edited model*: natural next-token KL, branch intervention error, and full-model group deletion effects on held-out documents and positions. Then compare uniform versus corpus weighting and weight, activation, ordinary-response, and intervention-aware legibility priors at matched rank and fidelity. The toy shows that sparse ordinary-response grouping can work when three rules are present; this real-model result specifies a necessary geometric correction before we can ask whether natural-language groups exist.

Reproduce with `flagship_rank96_function_grouping.py --n-queries 128 --n-keys 8192` and `rope_covariance_diagnostic.py`. The full split IDs, thresholds, graph sizes, edge retention, and per-gauge transport controls are in `full_width_results/head0_rank96_function_grouping_q128_k8192.json`.
