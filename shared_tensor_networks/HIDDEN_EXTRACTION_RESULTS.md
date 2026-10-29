# Hidden information versus usable transfer features

No source retraining. Fixed extraction on50stored models:15shared,15random-start independenth4,15released-from-shared,5teacher-start independent. Plan HIDDEN_EXTRACTION_PLAN.md fixed ranks and solvers before evaluation. Every original model/failure is retained. Four whole held-out tasks use the existing32/128noisy adaptation labels.

Extraction choices: ten source outputs; within-head hidden-product union (10shared or100independent features, no cross-head products); leading four normalized quadratic function directions followed by their ten products; or leading ten quartic union directions. SVDs are uncentered in the known Gaussian function metric. They use no held-out labels or teacher directions. The rank-four choice is the declared shared-architecture budget, not a rank discovered from data. SVD quadratics need not individually be bilinear: this tests a subspace, not neuron identification.

| Source model | Extraction | Median teacher transfer MSE,128labels | Median population projection floor |
|---|---|---:|---:|
| shared | outputs | 0.000851505 | 7.52507e-05 |
| shared | hidden_union | 0.000851505 | 7.52507e-05 |
| shared | quadratic_consensus | 0.000851505 | 7.52507e-05 |
| shared | quartic_pca | 0.000851505 | 7.52507e-05 |
| separate_width | outputs | 0.109226 | 0.086619 |
| separate_width | hidden_union | 0.186924 | 0.000102197 |
| separate_width | quadratic_consensus | 0.147431 | 0.0981327 |
| separate_width | quartic_pca | 0.156609 | 0.0975195 |
| released | outputs | 0.0343603 | 0.0232988 |
| released | hidden_union | 0.194502 | 1.50373e-05 |
| released | quadratic_consensus | 0.00121496 | 0.000627837 |
| released | quartic_pca | 0.00125208 | 0.000697795 |
| teacher | outputs | 0.0343603 | 0.0232988 |
| teacher | hidden_union | 0.191336 | 1.02683e-05 |
| teacher | quadratic_consensus | 0.00121496 | 0.000627837 |
| teacher | quartic_pca | 0.00125208 | 0.000697795 |

Each teacher entry averages all original initialization outcomes and four held-out heads. Teacher-start models have one outcome per teacher, not three independent replications. Every extraction agrees numerically for shared models: their feature spans are the same, providing a useful control.

## Every teacher: outputs versus quadratic consensus

| Source model | Teacher | Outputs | Four-quadratic consensus |
|---|---|---:|---:|
| separate_width | 225001 | 0.206812 | 0.153541 |
| separate_width | 225002 | 0.109226 | 0.147431 |
| separate_width | 225003 | 0.215404 | 0.201354 |
| separate_width | 225004 | 0.0502137 | 0.100337 |
| separate_width | 225005 | 0.0106614 | 0.0508078 |
| released | 225001 | 0.0343603 | 0.00121496 |
| released | 225002 | 0.101428 | 0.00113863 |
| released | 225003 | 0.0496789 | 0.00098969 |
| released | 225004 | 0.0209211 | 0.0314936 |
| released | 225005 | 0.00669109 | 0.0114277 |
| teacher | 225001 | 0.0343603 | 0.00121496 |
| teacher | 225002 | 0.101428 | 0.00113863 |
| teacher | 225003 | 0.049679 | 0.00098969 |
| teacher | 225004 | 0.0129291 | 0.00164841 |
| teacher | 225005 | 0.00586085 | 0.00255928 |

## Main qualification to the previous result

Releasing sharing did not erase all useful hidden information. Quadratic consensus reduces released-model median transfer MSE from.03436to.001215without additional source training or teacher-guided feature selection. Teacher-start independent models show the same median recovery. This remains worse than the shared median.000852and does not repair every failed shared initialization. Quartic rank-ten compression performs similarly, so this does not demonstrate a unique benefit of quadratic rather than quartic compression.

Random-start independent models are different: neither fixed compression reliably recovers their useful directions. Their100-feature union has a small population projection floor (.000102median), showing substantial representational coverage, but noisy128-label transfer is poor (.187median). Thus an expressive learned union is not automatically a sample-efficient transfer representation. The union has more adaptation parameters than the ten-feature methods.

## Exact conditional adaptation-risk decomposition

Let F contain feature-function coefficients, G their ambient Gaussian monomial Gram, X the adaptation feature design, and Y0 the noiseless adaptation targets. The fixed minimum-norm estimator with iid label noise E of variance sigma² is B=X⁺(Y0+E). For each task with coefficient vector c, conditional on the features and adaptation inputs,

E[population MSE] = ||F X⁺Y0−c||²_G + sigma² ||G^(1/2) F X⁺||²_F.

The first term includes finite-input approximation bias, not just the population projection floor. The second is pure adaptation-noise variance. This formula also applies to underdetermined designs with the declared pseudoinverse cutoff; it is not an assumption that quartic features themselves are Gaussian.

| Source model | Extraction | Median conditional bias | Median noise contribution |
|---|---|---:|---:|
| separate_width | outputs | 0.108096 | 0.000487159 |
| separate_width | hidden_union | 0.00146237 | 0.204963 |
| separate_width | quadratic_consensus | 0.146789 | 0.00055404 |
| released | outputs | 0.0342206 | 0.00052527 |
| released | hidden_union | 0.000171232 | 0.200977 |
| released | quadratic_consensus | 0.000769453 | 0.000512463 |
| teacher | outputs | 0.0342205 | 0.000549651 |
| teacher | hidden_union | 0.000149817 | 0.200977 |
| teacher | quadratic_consensus | 0.000769453 | 0.000519299 |

The independent union has noise contribution about.20, versusabout.0005for ten-feature extraction. This quantitatively explains why the union can represent the target yet transfer poorly with128noisy labels. The conditional expectation is an evaluation diagnostic, not a selection criterion used to tune extraction.

## Verification and scope

All50models/600projection or adaptation fits independently audited: source provenance,400fresh-input evaluation of quadratic/product features, normalization, leading-eigenspace equations, adaptation normal equations plus minimum-norm row-space condition, exact Gaussian errors and teacher-subspace evaluation.400conditional-risk decompositions replay actual noisy fits and agree between whitened andQRvariance formulas within1.39e-11. A naive Gram-trace calculation showed cancellation error up to.000153and was replaced by the stable factored formulas; that diagnostic is retained.

Sharing still improves practical learning in the earlier controls, but the output-span transfer gap overstates how much hidden information was destroyed by untying. Coherent feature extraction can recover much of it after a good shared/teacher start. The random-start extraction failures and the privileged Gaussian metric/rank budget limit broader claims. No native-model featurizer work was restarted.
