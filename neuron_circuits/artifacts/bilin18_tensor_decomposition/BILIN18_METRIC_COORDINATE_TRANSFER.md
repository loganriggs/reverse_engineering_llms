## H959 — Metric-coordinate prediction transfers across nouns, not reliably across templates

Trained ridge encoders to predict differences of H958's consistent shared budget1 feature states, with frozen .1/top8 output spans and both seeds. Features are either RMS-normalized raw activations or all64 frozen bilinear product codes. Fit differences, so arbitrary additive offsets of each quartet do not enter supervision. Scalar feature normalization uses training edges only; ridge alpha .001 fixed in advance. No held-out teacher coefficients or norms enter predictions, and predicted effects are not clipped using teacher norms.

Two separate protocols: seven complete noun-family holdouts (both templates held out for the noun), and two-way whole-template holdout (nouns shared across templates). All endpoint states in each evaluation fold are excluded from that fold's fit. The lexical protocol trains on both templates; it is not a joint lexical-plus-template novelty test. Targets remain teacher-informed, and holdout refers to this encoder training, not pretraining or all earlier research. 36 fits,4,704 live forwards. No backbone/span updates or held-out downstream selection.

Joint-path KLratio, ranges across seeds, normalized by omitting the MLP change:

| Factorial cell | Original top8 | Lexical holdout: normalized state | Lexical holdout: bilinear64 | Template holdout: normalized state | Template holdout: bilinear64 |
|---|---|---|---|---|---|
| prepositional:both | 0.849–0.875 | 0.518–0.679 | 0.543–0.671 | 0.985–0.999 | 1.771–1.969 |
| prepositional:distractor | 0.719–0.753 | 0.422–0.548 | 0.433–0.542 | 0.863–0.912 | 1.683–1.886 |
| prepositional:subject | 0.979–0.998 | 0.625–0.802 | 0.710–0.790 | 0.994–1.030 | 1.266–1.370 |
| relative_clause:both | 0.855–0.883 | 0.559–0.593 | 0.607–0.608 | 0.899–0.902 | 1.023–1.051 |
| relative_clause:distractor | 0.858–0.914 | 0.600–0.635 | 0.619–0.659 | 0.883–0.953 | 1.202–1.235 |
| relative_clause:subject | 0.688–0.773 | 0.453–0.495 | 0.504–0.512 | 0.774–0.834 | 0.778–0.852 |

Lexical holdout improves KL over the original encoder in every displayed cell/seed, with raw-state KLratios.422–.802 and bilinear-feature ratios.433–.790 versus original.688–.998. This shows that some oracle-coordinate gains can be predicted from new noun activations rather than requiring held-out teacher effects. Raw activation features are competitive with the frozen bilinear code; the latter is not necessary for these gains.

Generalization is incomplete: held-out coefficient errors are36–73% for raw state and44–76% for bilinear features, compared with training12–14% and25–27%. On template transfer, raw-state coefficient errors92–103% and bilinear104–188% despite lower training errors. Bilinear template-transfer KL is worse than omitting the MLP change in most cells (ratios1.02–1.97 except relative-clause subject .78–.85). Raw-state template transfer is near the omission baseline in many cases. This does not prove intrinsic nonlinear context dependence: limited data, distribution shift, fixed regularization, and inadequate input features remain alternatives.

KL improvement is not uniformly improved task-margin fidelity. For prepositional subject-only changes, lexical raw-state margin error54–82% can be worse than the original50–58%; the teacher-informed shared targets themselves favor the distribution metric over that selected margin. Absolute intervention KL remains small on this panel, as previously reported.

Audits: all36 saved encoders reproduce held-out coefficients exactly; complete endpoint/noun holdout assertions pass; normal-equation residual3.33e-14 and within-quartet consistency3.48e-16. The shared state-oracle comparator exactly replays H958 margins and KL. Native identity2.20e-5; exact joint historical margin7.29e-6; independent aggregate metrics<1e-12. All4,704 forwards completed, finite outputs, hashes unchanged. Reserved test untouched; natural document pool26,112. The development panel has been repeatedly analyzed; this is not pristine confirmation.

Interpretation: a small downstream-metric representation is partly learnable across lexical changes within represented contexts. We still lack evidence of a broadly reusable computational unit. Before increasing encoder capacity, determine whether label-conditioned prototypes explain the same gains.

Next: low-information controls with the identical folds and targets. Average centered state coordinates by template and subject/distractor number using training nouns only; compare a global number-state prototype for template transfer, and additive versus interaction prototypes. These controls receive explicit semantic labels and are not unsupervised featurizers. Their purpose is to reveal whether activation-conditioned prediction adds useful information beyond task/template labels, and whether joint-number interactions matter. Retain both protocols and seeds without tuning on downstream results.
