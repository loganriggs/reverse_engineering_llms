## H1015 — Recipient sensitivity is a largely aligned, number-selective functional readout

Previous turn was progress: H1014 showed response rank1transfers while activation-based factors remain inaccurate, with recipient gain a major limitation. This turn uses native gradients only as discovery mechanism diagnostics, not student inputs. All48recipients are H1013expanded-discovery examples.

Differentiate four contrast logits with respect to pre-layer11subject-position residual: is/are, was/were, can/will, period/comma. Exact native suffix includes attention, bilinear MLPs, normalization and softcap. Model weights frozen; only residual differentiated. The suffix's required four vocabulary contrasts are evaluated at the subject position; full-model comparison differs by1.43e-5absolute due to changed matrix-operation shapes. A central finite-difference check at+/-.001rho along each recipient's own source direction agrees with gradients to0.147%relative error.

| Discovery diagnostic | Singular | Plural |
|---|---:|---:|
| Rank-one share of raw is/are gradient energy |89.8%|90.2%|
| Median pairwise is/are gradient cosine |0.901|0.904|
| Gradient prediction error for quarter-scale odd response |0.37%|0.43%|
| was/were sensitivity relative to is/are |97.4%|86.0%|
| can/will sensitivity relative to is/are |4.9%|7.0%|
| period/comma sensitivity relative to is/are |4.7%|5.5%|

Sensitivities above are RMSgradient-projected responses across the same same-number donor directions; relative scales refer to this directional subspace, not global logit effects. The quarter-scale odd response is half the difference between positive and negative edit damage. Its accurate first-order prediction does not imply full positive edits are linear; H1013already measured substantial even response at full rho. Unrelated contrast checks here are local sensitivities, not full-scale behavioral collateral or capability tests. Related number contrast gradient cosine is0.928/0.938, consistent with a shared readout on this panel.

H1013response matrices were99.9%rank1in uncentered energy, while full read gradients here are~90%. This is consistent with extra gradient variation lying outside the tested source-direction family. It does not establish a universal one-dimensional computation. The perturbations all originate from one number-decoder intervention, so broad mechanistic claims still require other independently constructed interventions and output tasks.

Recipient gain from H1013finite-response SVD correlates0.915/0.889with gradient norm and0.913/0.974with mean source-direction sensitivity. Inverse input residual norm correlates only0.359/0.643; inverse final residual norm0.565/0.619. Native gradient-based quantities use the full suffix and cannot be substituted as free predictive features. Correlations are discovery diagnostics, not causal attribution of gain variation.

Final normalization test: detach only the final RMS denominator in backwards while preserving the clean forward. Clean contrast difference is3.34e-6absolute. The is/are gradient changes by26.9/26.3%of native gradient norm. This is a vector difference, not a fraction of independent causal effect; all earlier normalization, attention transport and output softcap remain live. It shows final normalization contributes but does not by itself explain recipient gain. The originally recorded all-four-contrast combined difference23.2/21.7%has a different denominator and must not be confused with is/are-only values.

Interpretation: a number-sensitive shared functional read direction with variable context gain is better supported than a generic amplifier or pure inverse-input-norm account, within this discovery intervention family. We still do not know a compact executable rule for that gain. This is a causal-target definition for supervised featurizers, not proof that activation manifolds or arbitrary neuron groups are the right units.

Audits: independent PyTorch projection/error calculation differs5.12e-17; all finite; native-gradient finite difference passed; no evaluation or reservedBSFtest states opened; no student training.48prefix sequences,48native sequences,192suffix sequences and384backward sequence-contrast operations. Saved native/frozen-denominator gradients, values, finite differences and raw/final states. No live process remains; natural-document pool26112unchanged. H961pending; broad goal active.

Next derive a shared functional direction from discovery gradients and use it to constrain an executable source/recipient predictor, compared against existing response-regression and activation-PCA controls. Charge gradients as discovery supervision, prohibit them at inference, preserve source/recipient split boundaries, and test both full-scale number effects and unrelated-output collateral. Before adopting any learned unit, require a second independent intervention family and multi-task transfer. Avoid further norm-only fitting or treating this one scalar target as resolution of the broader feature question.
