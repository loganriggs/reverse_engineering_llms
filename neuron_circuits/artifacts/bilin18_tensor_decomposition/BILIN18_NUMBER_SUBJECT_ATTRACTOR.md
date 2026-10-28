## H586 — The frozen direction is subject-dominant and survives irregular plurals, but is not a pure subject-number code

The previous turn made progress by identifying a compact causal direction and its limited scope. Freeze that direction and cross regular/irregular subject type, singular/plural subject, singular/plural intervening noun (the attractor), and prepositional/relative-clause frames. Examples: “The child beside the guards usually” and “The child that the guards can see usually”. Eight lexical quadruples per type/frame, 128 native prompts. The final token is always “usually”. Regular and irregular subjects share the same paired attractor nouns.

At Attention5's final write, separately swap only subject number or only attractor number. Test full, zero, frozen ranks1/4/8, nuisance and five random rank-one norm-matched controls: 2,816 interventions. Orient the frozen axis so its discovery plural-minus-singular mean is positive. Native factorial coefficient contrasts are descriptive; intervention output effects provide the separate causal test.

All eight type/frame/attractor-number competence cells pass. Both-endpoint candidate accuracy is 1.000 throughout; full-vocabulary intended-group top1 ranges 0.875–1.000. No performance-based exclusions. Lexical overlap with previous studies remains, but these crossed prompt combinations are new to this panel.

| Subject type/frame | Mean subject coefficient change | Mean attractor coefficient change | Full subject donor movement | Rank1 subject donor movement | Rank1 relative margin error | Rank1 relative reference KL |
|---|---:|---:|---:|---:|---:|---:|
|regular/preposition|607.5466|69.4764|2.3725|2.2753|0.0878|0.0275|
|regular/relative_clause|400.5838|98.7257|1.4938|1.1486|0.2721|0.1457|
|irregular/preposition|557.3313|76.6938|2.2782|1.9689|0.1390|0.0756|
|irregular/relative_clause|374.0972|97.9234|1.4805|1.0039|0.3219|0.1789|

All 64 subject singular-to-plural coefficient contrasts are positive, including irregular plurals with no regular -s rule. Subject main effects are about four to nine times attractor main effects in these cell means. This argues against a nearest-noun-only account and a regular-suffix-only account. It supports a subject-dominant number-associated signal at this operation/site.

But the attractor also shifts the coefficient: 63 of 64 attractor contrasts are positive. Its rank-one intervention generally moves the answer toward the changed attractor's number, even though the correct answer should remain governed by the subject. That effect is smaller than subject swaps but nonzero. The direction therefore is not an isolated, perfectly invariant subject-number variable.

Rank-one subject-swap fidelity varies. Regular prepositional prompts meet H584's joint <=10% margin/KL descriptive threshold (0.088/0.028). Irregular prepositional prompts miss the margin threshold (0.139/0.076); relative clauses miss both (regular 0.272/0.146, irregular 0.322/0.179). Capturing much of the mean donor movement is not full-swap equivalence. Rank eight improves some errors but does not remove the relative-clause gap. Attractor-swap approximation is much poorer, and it is not an independent additive decomposition.

There are token-length changes in some singular/plural pairs (recorded per quadruple), so morphology, lexical identity and position are not fully separated. More fundamentally, the main subject is always the first noun so far. Subject dominance could still be a first-noun heuristic rather than grammatical-role tracking. Irregular subjects also differ in animacy and lexical distribution; this is not a controlled morphology-only comparison.

Next H587 moves the prepositional phrase to the front: “Beside the guards, the child usually”. This puts the attractor first and grammatical subject second, with the frozen direction and all crossed numbers retained. Compare coefficient main effects and full/projected causal responses; preserve competence failures. Following the grammatical subject across both noun orders would weaken both a first-noun-only and a nearest-noun-only explanation, without proving universal syntax abstraction.

Audits: zero replay <=1.34e-5 margin; an independent direct-overwrite run reproduces all 256 full swaps within 1.34e-5; source/basis/manifest unchanged; finite metrics. No new natural-text documents; FineWeb total 26,112. Broad goal active and incomplete.
