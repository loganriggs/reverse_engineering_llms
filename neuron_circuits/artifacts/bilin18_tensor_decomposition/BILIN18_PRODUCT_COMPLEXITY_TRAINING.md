## H949 — A causal contribution penalty concentrates learned responses, with a fidelity tradeoff

Question: can a preference for fewer whole bilinear products turn the previously distributed anchored response into a compact explanation?

Design: eight fits, two seeds and four predeclared penalty weights (0, .01, .1, 1), each initialized from the same-seed H946 effect-anchor fit. All use 64 products, identical parameter count, 2,000 steps and matched natural minibatches. The base objective preserves natural reconstruction/random-probe effects and the 15 discovery anchor pairs. Added cost is the sum of smoothed quarter powers of each product's squared finite-response contribution, normalized by teacher response energy and averaged equally over the two tasks. This charges entire contributions, is invariant to factor scale and compensated signs, and penalizes splitting an identical contribution into multiple copies. It is not a global storage or exact active-product penalty.

Results on existing held-out full interventions (relative vector-norm error; lower is better):

| Penalty | Agreement: top-8 error vs complete student | Sentiment: top-8 error vs complete student | Agreement: all-64 error vs teacher | Sentiment: all-64 error vs teacher |
|---|---|---|---|---|
| 0 | 81–84% | 64–70% | 72.5–72.7% | 56.5–56.6% |
| .01 | 33–45% | 21–24% | 72.7–73.0% | 56.7–56.8% |
| .1 | 23–24% | 13.6–13.7% | 73.5–73.7% | 58.2–58.3% |
| 1 | 84% | 67–81% | approximately 100% | approximately 100% |

The .1 penalty's top eight products have teacher errors 73.3–73.6% agreement and 57.8–58.0% sentiment: concentration is not solely a relative metric artifact. With that penalty, one sentiment product already approximates the complete student within 25% error in both seeds. Agreement reaches that threshold with 2 or 8 products depending on seed. These are discovery-greedy selections, not optimal subsets; selected counts are limited to the tested grid. Subset errors need not be monotonic because contributions can interfere.

Interpretation: the distribution across many products was partly adjustable through the training objective. A moderate contribution cost substantially concentrates the anchored response while modestly worsening already-incomplete teacher fidelity. The largest penalty instead suppresses the response. This supports a compactness/fidelity tradeoff, not discovery of unique, semantic, or reusable computational units. Natural random-effect fidelity remains poor (roughly 102% error at .1); these models are not general MLP substitutes. The same repeatedly used development panel is not a fresh confirmatory test.

Verification: all eight checkpoint predictions replay exactly. Independently materializing every product's output contribution agrees with the factored energy formula to 1.82e-7 relative error. Factor rescaling and compensated-sign changes preserve the cost exactly in the tested checks. Analytically splitting a contribution in two increases the smoothed penalty by about 1.41x. Input hashes unchanged; finite outputs; zero whole-model forwards in this experiment; reserved test untouched; natural document pool remains 26,112. Full downstream output fidelity and cross-seed component correspondence remain untested for these fits.

Next: test both complete fits and discovery-selected top-eight contributions in the live downstream model, retaining every penalty and seed. Compare isolated MLP and joint residual-plus-MLP intervention effects, full-distribution KL, and zero-change baselines. Only then assess whether the concentrated components transfer and match across seeds.
