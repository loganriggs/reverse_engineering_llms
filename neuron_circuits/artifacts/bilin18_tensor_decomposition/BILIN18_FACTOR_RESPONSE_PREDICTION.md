## H1014 — Separable causal response transfers; activation-based gain prediction remains incomplete

Previous turn was progress: H1013 separated source magnitude/direction/context and found nearly separable scalar responses. This turn fits source-potency and recipient-gain predictors on discovery, freezes their predictions before loading the model for evaluation, and collects new Cartesian interventions. No suffix outcomes are supplied at student inference.

For each grammatical number, rank-one SVD of the24x24discovery full-positive-rho matrix yields gain(normalized to mean1) and potency. Predict potency from1152unit-source-direction coordinates or their training-PCA4; predict gain from six clean-source/residual norms or clean residual PCA4. Each centered kernel is normalized to mean diagonal1and ridge0.1is fixed. Separate per-number models use the known benchmark number label. Include constant and source-only controls. This is24training source vectors and24recipient vectors per number, not576independent feature-training examples. Frozen model coefficients and all predicted matrices were saved before evaluation suffix calls.

Evaluation: eight noun pairs from H1012test pool, donor directions from near-context examples;16recipients per number (eightnear,eightunder). Each of8directions is transplanted at fixedrho1803.343into each recipient. Four equal-norm random directions are controls. These noun states were previously opened in H1012evaluation, so this is new intervention-combination evidence, not untouched lexical data. Native prefixes supply source/recipient states; model full suffix is used only to generate reference outcomes.

| Predictor | Singular full-matrix relative L2error | Plural full-matrix relative L2error |
|---|---:|---:|
| Constant |38.8%|39.8%|
| Source linear, recipient mean |24.3%|42.5%|
| Source linear x recipient norms6 |17.0%|41.6%|
| Source linear x recipient PCA4 |21.1%|28.2%|
| Source PCA4 x recipient PCA4 |22.8%|28.6%|

All variants reported; there is no uniform winner and no post-evaluation model adoption. Training errors for source-linear/recipient-norms were10.4/12.0%; with recipientPCA4they were8.8/11.9%. Finite-scale response prediction still has substantial generalization error. H1011/H1012scores use different variable-magnitude joint-source edits; do not directly compare their percentages with this fixed-norm Cartesian target.

The low-rank RESPONSE structure does transfer: evaluation-answer rank-one oracle errors2.01%singular/2.13%plural. Task-derived versus random equal-norm damage RMS ratios13.45/13.78. This separates target compressibility from ability to predict its coordinates. A low-rank oracle is not an activation-based explanation.

Post-result factor-replacement diagnostic (uses evaluation answers and is not deployable): keeping the learned linear source potency while solving optimal recipient gains yields10.44%singular/5.79%pluralerror. Keeping PCA4recipient gains while solving optimal source potencies yields10.04%singular/13.26%plural. For plural, measured recipient gains rescue the largest part of the error; source potency generalizes better up to a rescaling. Norm-based recipient gains are particularly weak there (even optimal source factors leave29.57%error). Both factors and their scale calibration remain imperfect; their independent scale is gauge-dependent, so these tests identify conditional prediction limitations rather than unique physical quantities.

Audits: native margin replay exact0; frozen prediction hash unchanged; independent PyTorch evaluation of saved source/recipient models reproduces all predictions to8.89e-16absolute; training/test source and recipient row overlap0;finite.32native sequences,32prefix sequences,416suffix sequences. No reservedBSFtest data or new natural documents; pool26112unchanged. No live job remains. Broad goal active; no featurizer/manifold claim, H961pending.

Next investigate recipient gain using discovery-only mechanistic diagnostics: determine whether normalization, downstream read direction, or context-dependent transport controls the gain, before another feature search. Compare functional read directions across recipients with native gradients/finite differences strictly as diagnostics, charging their native computation; then derive an explicit predictor that does not use those diagnostics at inference. Add other output contrasts to distinguish number-selective sensitivity from a generic output amplifier. Only after that freeze new cross-context/multi-task tests. Do not redefine success as predicting this one deliberately rank-one-initiated intervention.
