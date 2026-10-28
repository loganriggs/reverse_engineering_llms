## H979 — A single-state compact quadratic readout predicts scalar causal edits on the development panel

Previous turn was progress: H978 derived compact output-aware factor readouts but still required evaluation quartets to extract main-factor inputs. H979 removes that prediction-time dependence. Each model accepts one normalized adjective-position MLP3 input x and fixed parameters only. No evaluation labels, quartet means, donor outputs or batch statistics enter scalar prediction. Donor/recipient pairs are used afterward to evaluate counterfactual differences.

Use the frozen discovery output direction u and its exact symmetric bilinear scalar form F. Discovery mean mu comes from the original16 states. Let Q span the discovery negation directions N and their weight-derived readouts F*N. The predictor is c + (F*mu)^T(x-mu) + 0.5*z^T(Q^T F Q)z, z=Q^T(x-mu), c=0.5*mu^T F mu+u^T Down_bias. Negation rank4 gives8 quadratic coordinates, plus one full linear readout. Count nine scalar input readouts unless the linear direction happens to lie inside Q; do not describe this as only eight total features. Rank1 negation gives two quadratic coordinates. Controls use discovery PCA8, random8, a linear-only tangent and an uncentered aware8 predictor.

Reference intervention differs from H977/H978: for all12 directed edges in each quartet, add u times the exact donor-minus-recipient MLP scalar difference at the recipient adjective position. Candidate intervention replaces that scalar difference with differences of independent predictions. This tests one output-direction counterfactual, not full MLP replacement or the prior local-interaction-only edit. KL ratio is discrepancy to exact scalar-edit output divided by no-edit discrepancy; zero is exact, one is no edit. Percentages must not be directly compared with earlier local-removal benchmarks.

| Native-passing cohort | Centered aware8 | Discovery PCA8 | Linear only | Uncentered aware8 |
|---|---|---|---|---|
| Film anchors |0.0011|0.0091|0.5922|0.2737|
| Movie anchors |0.0011|0.0123|0.6011|0.2636|
| Book anchors |0.0013|0.0133|0.5847|0.2718|
| Film new3 |0.0124|0.3099|0.6757|0.3361|
| Movie new3 |0.0139|0.3314|0.6832|0.3251|
| Book new1 |0.0059|0.0855|0.6283|0.1404|
| Book new2 |0.0332|0.2284|0.7095|0.1916|
| Book new3 |0.0155|0.3093|0.6896|0.3461|

Random8 is approximately the linear baseline; awareness of the weight/output relation matters beyond rank alone. Centering with the exact linear term also matters. For book/new1, edge-specific aware8 KL ratios are0.0010 for negation toggles,0.0068 for adjective changes and0.0517 for both-factor changes. PCA is slightly better on the last subset (0.0460), so aggregate performance does not imply dominance on every edit type. Both-factor edits preserve ideal binary sentiment and expose residual distinctions.

This is the first candidate in this branch that predicts from individual activation states without evaluation quartet factors. It is a supervised, weight-informed scalar computational readout, not an unsupervised feature ontology, a full-layer featurizer, or proof of an intrinsic nine-dimensional model representation. Fits use original discovery data only, but architecture choices were informed by repeatedly inspected development results. A fresh prospective panel is needed before claiming independent generalization. The frozen output direction supplies a specific causal objective, illustrating a useful prior for future featurizer training.

Audits:6,144/6,144 whole-model forward examples. Native identity logit error2.19e-5. Exact scalar weight replay maximum absolute error0.0122 against teacher scalar RMS7036 (float32 capture versus double algebra). Predicting individually versus batched differs at most5.46e-12; independent NumPy implementation differs1.10e-11. Frozen models/input hashes unchanged, summary aggregation independently checked, outputs finite. No gradient training or original reserved-test access. Natural-document pool26,112.

Next: lock the actual single-state model artifact and evaluate it on an additional prospective lexicon/context panel without modifying parameters or ranks. Include the frozen scalar reference, PCA and linear controls and all competence failures. Then test whether this representation remains selective under nuisance changes and can support the earlier local-composition/rescue findings, rather than assuming scalar counterfactual fidelity establishes every mechanistic claim. H961 frozen context-bridge predictor evaluation remains pending.
