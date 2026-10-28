

## H314 — Explicit cross-input kernel versus additive input predictor

For each18 head/payload group fit kernel ridge on3072 reference sites from first768 discovery documents, predicting cached missing cross responses after H300 refined64. Next128 documents select one ridge per family/kernel from.01/.1/1/10; final128 evaluate. All are reused discovery documents, with disjoint marginal calibration/evaluation. No intercept. Same targets and reference count for both representations, but different feature dimensions/inductive biases; not equal parameter capacity.

Let u be projection of centered state onto frozen core96 and v its component orthogonal to core96 and partner64. Interaction feature is phi(x,z)=u_z⊗v_z−u_x⊗v_x. Its kernel is evaluated by four products of inner products, without materializing the full tensor. Additive features concatenate midpoint and difference of u and v, each block normalized by reference mean squared norm. Each complete kernel is normalized to reference mean diagonal1. The interaction kernel is a linear model in the exact bilinear cross-product representation, so its advantage would support the relevant inductive bias, not discovery that this known bilinear model has interactions. Native output coefficients are not used in the fitted readout; targets remain native-supervised.

| Family | Kernel | Selected ridge | Calibration fidelity | Evaluation fidelity |
|---|---|---:|---:|---:|
|actual|interaction|0.01|42.78%|42.22%|
|actual|additive|1.0|10.99%|10.09%|
|angle|interaction|0.01|24.53%|23.04%|
|angle|additive|10.0|2.18%|2.22%|

Interaction minus additive evaluation gains (post-result paired document bootstrap, conditional on frozen selection):

- actual: 32.13pp,95%[31.35,32.95].
- angle: 20.83pp,95%[19.00,22.65].

Locked gates: interaction_gain: PASS, interaction20: PASS. Interaction gain requires>=5pp in both families; coverage requires>=20% evaluation missing-response fidelity in both. Bootstrap intervals are supplementary, not added to locked gates.

The selected interaction model recovers42.22% actual and23.04% angle missing-response energy, versus additive10.09%/2.22%. This is a substantial change from the calibrated nearest-neighbor direction results, and supports explicit interaction structure as a useful prediction representation. Both interaction families select the smallest tested ridge(.01); no ridge optimum or estimation ceiling is established. The additive choices are1(actual) and10(angle). Fresh confirmation should retain these choices without extending the search on the new panel.

**Scope and cost.** This is a representation diagnostic, not a compact compiled model. A naive dual readout stores3072×1152 coefficients PER group (~3.54M),18 groups ~63.7M, excluding reference descriptors. Interaction features have at most96×992 independent coordinates after excluding the160-dimensional core/partner span, versus additive midpoint/difference features of at most2176 coordinates. Kernel rank is limited by3072 examples. Thus a large advantage cannot be attributed to multiplicative structure independently of capacity without further controls. No cross-head parameter sharing, fresh confirmation, intact substitution or downstream KL test is claimed.

**Audit.** Small explicit outer-product features replay the four-term kernel within relative1e-10. Symmetrized training Gram spectra are PSD within1e-7 tolerance; tiny negative numerical eigenvalues are clamped to zero before ridge. All kernel mean diagonals are logged. Summary evaluation fidelities replay from saved document-level errors. Source response hash checked; all jobs terminal, original native model/compiler/base unchanged.

**Next discriminator.** Freeze the selected predictor and establish fresh transfer before optimizing storage. For a positive result, compare shared cross-head readouts and rank-restricted factorizations with matched data and explicit cost. For a weak result, use learning curves and oracle coefficient accounting to separate underdetermined estimation from representation error; do not equate kernel regression failure with absence of bilinear structure. Broad research goal active.
