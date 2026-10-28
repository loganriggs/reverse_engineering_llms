## H960 — Label prototypes explain much of metric-coordinate transfer

Same lexical and template holdouts as H959; same frozen .1/top8 spans and H958 shared budget1 targets. Controls receive explicit subject number, distractor number, and template labels, but no activations. Average centered training-noun states to form four-state prototypes. Compare template-conditioned versus pooled prototypes, and full versus additive subject+distractor decompositions. Template holdout uses a prototype from the training template only. No held-out teacher values/norms enter prediction. Privileged labels make this an explanatory control, not an unsupervised featurizer.

Lexical holdout, joint-path KLratios (ranges across seeds):

| Factorial cell | Template prototype, full | Template prototype, additive | Activation encoder (H959) |
|---|---|---|---|
| prepositional:both | 0.567–0.730 | 0.567–0.730 | 0.518–0.679 |
| prepositional:distractor | 0.449–0.579 | 0.534–0.638 | 0.422–0.548 |
| prepositional:subject | 0.695–0.882 | 0.860–0.974 | 0.625–0.802 |
| relative_clause:both | 0.592–0.636 | 0.592–0.636 | 0.559–0.593 |
| relative_clause:distractor | 0.630–0.665 | 0.631–0.690 | 0.600–0.635 |
| relative_clause:subject | 0.485–0.528 | 0.483–0.515 | 0.453–0.495 |

The raw-activation encoder improves joint KLratio over the full template prototype in all12 cell/seed comparisons, but only by.027–.080. Therefore much of H959's gain is compatible with template/number-conditioned response prototypes; there is a smaller additional activation-dependent benefit. This is not evidence that the model literally uses these labels or stores these prototypes. The labels are supplied by the experimenter, while the activation encoder infers its prediction from state vectors.

Pooling templates worsens fidelity substantially, especially for relative clauses. Template-transfer label prototypes often fare worse than zero MLP change: relative-clause full-prototype KLratios1.32–2.48, and prepositional subject-only1.41–1.54. Context-specific response patterns matter; simply naming the same subject/distractor variables does not align the output behavior across these templates. Limited training support, not an intrinsic impossibility of shared computation, remains a plausible explanation.

A decisive design lesson: full and additive prototypes are exactly identical on both-number flips. With signed factors h,d in{-1,+1}, the interaction hd is unchanged under (h,d)→(-h,-d). Such contrasts cannot test the interaction contribution. Independent subject-only and distractor-only flips expose it. On prepositional subject changes, removing the interaction worsens KLratio from.695–.882 to.860–.974; on distractor changes from.449–.579 to.534–.638. Relative-clause gains from the interaction are small or inconsistent, and additive subject-only prototypes are slightly better in KL. These are interactions in the fitted finite-state representation, not proof of a unique mechanistic product.

Task margins and KL again disagree: additive prepositional subject prototypes improve the task-margin error while worsening KL. Avoid promoting components using only one selected task score. Absolute intervention KL remains small on this panel.

Audit:4,704/4,704 live forwards; native identity2.20e-5; exact joint replay7.29e-6; centered-state and additive/interaction algebra errors3.42e-13; loop consistency2.52e-16; independent saved-prototype replay2.62e-8 relative; both-flip interaction cancellation exact. Training/evaluation endpoints and groups independently checked disjoint; aggregate metrics independently recomputed. Inputs unchanged, finite results, no backbone/span training or reserved-test access. Natural document pool26,112.

Interpretation: we have a small supervised, context-sensitive description of causal output changes with some lexical transfer. The evidence does not yet support a general recovered unit or meaningful unsupervised manifold. Current strongest alternatives are (1) template-conditioned prototypes plus lexical corrections; (2) a shared computation expressed in context-dependent coordinates; (3) an underdetermined encoder that fails extrapolation because training contexts are too sparse; and (4) context-dependent downstream sensitivity creating different quotient representations of the same upstream information.

Next: stop tuning this two-template panel. Freeze encoders/prototypes and construct prospective context bridges with fixed input-only choices: nearby prepositional variants, nearby relative-clause variants, and changed distractor lexemes, with complete number factorials and native competence checks. Compare generalization as input geometry and syntax vary. This will separate small within-family shifts from across-family shifts before spending more effort on a richer encoder or calling the groups manifolds. Include failure cases and zero-change/full-distribution controls; preserve the original reserved test.
