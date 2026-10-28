## H977 — Compact output transfers better than discovery PCA input spans

Previous turn was progress: H975/H976 supplied prospective competence and frozen-site intervention/rescue transfer. H977 fits adjective and negation input spans only from original four familiar film quartets at normalized adjective MLP3 input. SVD rank1 and rank4 bases are frozen before evaluation; rank4 spans each set of four discovery main-factor vectors. Project new quartet a,n vectors, compute the MLP's exact bilinear local cross term, and compare its removal against full teacher local-term removal. Also test restricting only one input side, teacher-output projection onto discovery output spans, and three seeded dimension-matched random input-span pairs. No norm rescaling.

KL ratio = summed KL(teacher-local-removal output||candidate-removal output) / summed KL(teacher-local-removal output||native output). Zero reproduces the full teacher intervention; one is the no-edit baseline. All64 quartets evaluated, with native-gated cohorts distinguished.

| Native-passing cohort | Both input spans rank4 | Negation-only rank4 | Output rank1 teacher-projection baseline |
|---|---|---|---|
| Film anchors |0|0|0.0059|
| Movie anchors |0.0019|0.0001|0.0051|
| Book anchors |0.0015|0.0014|0.0052|
| Book new1 |0.1995|0.0118|0.0109|
| Book new2 |0.2649|0.0469|0.0080|
| Book new3 |0.5816|0.0941|0.0458|
| Film new3 |0.5585|0.1005|0.0446|
| Movie new3 |0.5741|0.1045|0.0442|

Random input rank4 spans give KL ratios approximately1. Discovery anchor reconstruction is expected because rank4 spans all fitted factors; it is an implementation check, not generalization evidence. Changing review noun with familiar adjectives transfers well. New lexical pairs expose the limitation: projected adjective factor retains only about9–33% of input squared norm, versus65–85% for negation factors. Both-input local-output L2 relative error is about47–73% on new native-passing cohorts. Restricting adjective alone is substantially worse than restricting negation alone.

A frozen rank1 output direction retains much greater downstream fidelity, but its coefficient is obtained by projecting the full teacher local output. This is a privileged span-adequacy baseline, not a predictive compact operator. It shows a reusable result direction does not imply that discovery PCA input directions capture the computation. Conversely, PCA failure does not prove the task-relevant input is intrinsically high-dimensional: large lexical input variation may be irrelevant to the output direction.

A compact4x4-to1152 bilinear operator constructed from frozen input bases exactly reproduces the projected-input calculation (relative contraction error6.03e-16). The missing fidelity comes from the input restriction, not operator implementation. No state-based featurizer is established: even input projections use quartet main-factor vectors, not one independently encoded state.

Audits:2,816/2,816 whole-model forward examples; frozen basis hashes and source hashes unchanged; identity logit error2.19e-5; recomputed teacher local terms match H976 within2.66e-8 relative; teacher-removal margins replay within1.27e-5. Independent NumPy discovery projectors agree3.07e-16; bases orthonormal within8.89e-16; aggregates checked, finite outputs. No gradient training, site retuning or reserved-test use; natural-document pool26,112. SVD bases are explicitly fitted on discovery labels, not claimed unsupervised.

Next: test output-aware input directions using the actual bilinear weights and frozen discovery output direction. For scalar output u, its symmetric bilinear form F maps negation basis U_n into adjective readout directions F*U_n. These directions can preserve the output-relevant contraction without preserving all lexical input variation. Compare this weight-informed compact operator with PCA spans and the privileged output-projection baseline on the same locked panel; then assess state-based encoders on an additional prospective panel rather than claiming deployment from quartet adequacy. H961 frozen context-bridge predictor evaluation remains pending.
