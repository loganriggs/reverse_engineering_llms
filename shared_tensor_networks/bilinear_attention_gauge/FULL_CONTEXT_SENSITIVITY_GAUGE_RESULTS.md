# Full-context sensitivity gives a sparse exact gauge, but axes are not canonical

Earlier first-layer pair-response gauges found sparse or token-form-associated
directions, yet some proposed directional labels failed a full-model input-edit
test. This experiment uses a downstream behavioral signal to fit the gauge:
the gradient of the **last-position next-token log probability** through all
18 layers with respect to each first-layer pre-RoPE QK1 key. The key gradient
is projected into the already fitted rank-256, nine-head function-aware
dictionary. Every compared gauge is an orthogonal rotation of that one fixed
dictionary; all of them compute exactly the same reduced model function.

For each 64-token context, one backward pass yields 64 key-position gradient
vectors. Fit a varimax rotation to row-normalized vectors from 64 FineWeb
training documents (4,096 key positions). Validation uses 24 disjoint
FineWeb documents, and tests use 24 further FineWeb documents plus 24 each
from WikiText-103 and TinyStories. Neither test corpus nor test document
helps choose the rotation. Controls are the supplied dictionary coordinates,
a Haar rotation, weight-varimax, and an earlier pair-response-varimax gauge.
The full-context fit and two random-start refits each use 240 varimax steps.
The objective is still improving slowly at that cap, so this is a
fixed-compute comparison rather than a convergence claim.

The primary descriptive metric is the mean number of effective coordinates
per key-position gradient, `(L1 norm)^2 / (L2 norm)^2`. Smaller means the
same downstream sensitivity is carried by fewer axes. Because each row is
scored equally, a few high-gradient documents cannot directly dominate
this metric. A paired document bootstrap gives uncertainty for differences.

| Gauge | FineWeb test | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Supplied function-aware coordinates | 137.7 | 141.2 | 139.0 |
| Haar rotation | 163.9 | 163.7 | 164.0 |
| Weight varimax | 131.7 | 136.4 | 134.1 |
| Pair-response varimax | 149.7 | 151.2 | 149.3 |
| **Full-context sensitivity varimax** | **96.2** | **105.0** | **97.8** |

The full-context-minus-supplied differences are `−41.5` on FineWeb
(95% document-bootstrap interval `[−44.6, −38.3]`), `−36.2` on
WikiText (`[−40.3, −31.9]`), and `−41.2` on TinyStories
(`[−43.5, −38.8]`). All 24 documents per corpus favor the full-context
rotation. The two independent random-start fits give approximately
`97.2/105.8/98.4` and `97.2/105.7/98.7` in corpus order. Thus the
*aggregate sparsity* transfers, even though individual axes are not fully
identified.

After optimal sign/permutation matching, independent starts have mean
absolute axis cosines around `0.63–0.64`; about 53–55% of matched axes
exceed `0.8`. This is much stronger than arbitrary rotations but leaves
many unstable coordinates. A single named direction should therefore not
be treated as canonical without cross-fit and causal tests.

As a limited legibility proxy, select each gauge's 20 highest-gradient-energy
axes on training documents and assign each the enriched **key-token form**
among seven broad categories (punctuation, space-prefixed word, and so on).
On test documents, score the unchanged category after normalizing each
key-position gradient to unit total energy. Mean held-out log2 enrichment
for the full-context gauge is `0.35/0.38/0.17` on FineWeb/WikiText/TinyStories,
versus `0.15/0.04/0.24` for pair-response varimax. This proxy is mixed:
full-context fitting helps on two corpora, but not TinyStories, and these
categories are not task labels. Raw, unnormalized gradient-energy enrichment
is especially fragile: the gradient-energy effective document counts are
only `6.3`, `4.8`, and `2.2` on the three test corpora, with the largest
TinyStories document carrying 66% of energy. Row-normalized label scores
are the more appropriate exploratory comparison here.

The custom first-layer hook exactly matches the existing rank-256
projection on a checked head-output batch. A central finite difference
of one key-direction perturbation matches autograd for the next-token
log-probability derivative to `1.9e-8` absolute error. These checks validate
the sensitivity implementation, not the interpretation of any axis.

**Decision.** Full-context downstream sensitivity is a useful prior for a
sparse coordinate system on this fixed reduction. It does not yet give a
canonical decomposition of the attention function or a selective causal
mechanism. The next test should choose axes that recur across independent
fits, use matched-size *finite* key-direction interventions in new contexts,
and require them to predict specific held-out edit or task effects against
same-gauge and random-axis controls. In particular, a concentration gain
alone should not be used to name a linguistic computation.

Reproduce with `flagship_full_context_sensitivity_gauge.py --train-docs 64
--test-docs 24 --length 64 --seed 1031`, then
`analyze_full_context_sensitivity_gauge.py
all_head_full_context_sensitivity_gauge_train64_test24_l64_s1031`. The
per-position gradients, keys, and rotations are in the companion NPZ under
`full_width_results/`; the JSON and `_analysis.json` hold labels, paired
document intervals, and independent-start axis matching.
