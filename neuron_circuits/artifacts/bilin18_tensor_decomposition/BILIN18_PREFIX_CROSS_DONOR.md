## H620 — Country-independent complement suppresses answer emission while preserving eight-way capital identity

The previous turn was progress: H619 showed that the old prefix-absence direction is unusually damaging even after matching another prefix interaction's magnitude. H620 tests whether that direction is tied to country identity. Deduplicate H618 combined prompts before intervention:8 unique countries in each of two templates,16 copying names and16 agreement endpoints from template0. For each recipient, subtract every same-template, same-token-length donor's complement to frozen H615 joint64, scaled to the recipient complement's total norm. Doses0.25 and1 include self removal, native replay, and one seeded norm-matched random-complement control per recipient. No new basis is fitted.

This yields1,364 evaluations including48 native replays. Each capital template has56 cross-country ordered donor/recipient pairs per dose. Copying has240 cross-donor pairs; agreement has210 because token-length matching splits that panel. Counts are repeated interventions on a small set of prompts, not independent samples. No output-based filtering occurs.

**The harmful component transfers across countries.** At dose1, cross-country removal has nearly the same mean output KL, answer mass and generic-token competition as self removal. Correct capital remains first among all8 candidate capitals in every capital intervention, including self, cross-country and random controls at both doses. However, unrestricted correct top1 falls sharply under self and cross-country removal.

| Capital template | Intervention | Dose | Output KL | Correct full-vocabulary top1 | Correct among8 capitals | Correct capital mass | Correct conditional on8 capitals | Capital vs generic margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
|0|native|0.0|0.0000|1.0000|1.0000|0.3604|0.9733|1.4018|
|0|self|0.25|0.0277|1.0000|1.0000|0.2952|0.9722|0.9011|
|0|donor|0.25|0.0280|1.0000|1.0000|0.2950|0.9722|0.8998|
|0|random|0.25|0.0006|1.0000|1.0000|0.3596|0.9728|1.4078|
|0|self|1.0|0.8969|0.1250|1.0000|0.0596|0.9443|-1.4445|
|0|donor|1.0|0.9057|0.1429|1.0000|0.0598|0.9429|-1.4554|
|0|random|1.0|0.0098|1.0000|1.0000|0.3429|0.9695|1.3369|
|1|native|0.0|0.0000|1.0000|1.0000|0.2645|0.9715|0.5827|
|1|self|0.25|0.0281|1.0000|1.0000|0.1999|0.9675|0.0944|
|1|donor|0.25|0.0284|1.0000|1.0000|0.1995|0.9674|0.0922|
|1|random|0.25|0.0004|1.0000|1.0000|0.2646|0.9715|0.5765|
|1|self|1.0|0.5866|0.1250|1.0000|0.0431|0.9342|-1.8383|
|1|donor|1.0|0.5912|0.0893|1.0000|0.0429|0.9332|-1.8491|
|1|random|1.0|0.0064|1.0000|1.0000|0.2559|0.9707|0.4984|

At full dose, cross-country correct top1 is14.3%/8.9%, versus12.5%/12.5% under self removal and100%/100% for random controls. Mean KL is0.906/0.591 versus0.897/0.587 for self and0.0098/0.0064 for random controls. The correct capital's mean probability drops from0.360/0.265 to0.060/0.043, whereas its conditional probability among the eight capitals remains0.943/0.933. That conditional probability does fall somewhat from0.973/0.972; “identity preserved” means rank preservation here, not exactly unchanged factual probabilities.

Capital-vs-generic margin compares the correct token with the combined probability of "the", "a", "not", fixed from H618. It flips from+1.40/+0.58 natively to−1.46/−1.85 under cross-country removal. At quarter dose, probability mass and margins already shift, but every full-vocabulary capital answer remains correct. These results describe graded distribution changes before accuracy crosses a threshold.

**The donor directions are already almost identical.** Within each template, flattened body-write complement cosine ranges0.9922–0.9986 and0.9892–0.9982. Successful transfer therefore supports a common scaffold/context component on this panel; it is not evidence of transporting distinct country-specific facts between unrelated coordinates. Norm matching is global over all body positions; per-position magnitudes are not separately matched.

**There is little donor-specific capital preference beyond the common effect.** A donor/recipient log-odds shift alone would be misleading: at dose1 the donor capital gains about1.19 nats relative to the recipient capital on average, but other wrong capitals can gain similarly. We therefore inspect all eight capital log probabilities. The donor's change relative to the mean log probability change of the other six wrong capitals averages−0.0036/−0.0031 nats. Relative to the corresponding self-removal contrast, per-pair changes range−0.060 to+0.050 and−0.041 to+0.041 nats. This weakens a donor-identity-content explanation. It does not prove exact identity independence; small effects remain, and the panel is finite. Token log probabilities were added to the result schema in a second identical run to permit this audit; conditions and donor selection were unchanged.

Copying remains correct throughout, with negligible complement effects. Agreement remains correct throughout this selected template; donor KL is about0.0063 at dose1, versus0.0012 for random controls. These are within-family transfers, not cross-task transplantation of capital directions. They do not establish a shared semantic unit across tasks.

Twenty competing hypotheses, refreshed after the last five experiments:

1. A shared component spans countries within these capital templates — strongly supported locally.
2. The harmful complement primarily stores donor-specific capital identity — weakened by cross-country transfer and selectivity audit.
3. It controls direct-answer emission versus generic continuation — supported as an operational description on these prompts.
4. Capital identity probabilities are completely unchanged — false; conditional probabilities shift despite preserved rank.
5. High correct-answer accuracy implies internal fidelity — contradicted by earlier copying results.
6. Correct paired choice implies full factual preservation — insufficient; eight-way checks strengthen but do not universalize it.
7. Any equal-magnitude perturbation causes answer suppression — weakened by random controls and H619.
8. The effect is only a large-dose artifact — weakened by quarter-dose probability changes; accuracy failures remain dose-dependent.
9. Nonlinearity alone explains poor frozen-space transfer — weakened by small-dose complement effects.
10. Finite-dose nonlinearity contributes — supported by H617.
11. The copying-trained span is task-general — weakened by H618.
12. A universal low-dimensional semantic basis has been found — unsupported.
13. Data coverage determines useful compact coordinates — supported across H614–H618.
14. Internal Euclidean error predicts output KL monotonically — weakened by rank/objective reversals.
15. A response-distribution quotient may be smaller than an internal-vector representation — plausible, conditional on distribution and tolerance.
16. Prefix-absence/lowercase context is especially relevant — supported by H619, though direction and context remain coupled.
17. Initial-token position effects explain the common direction without semantic content — open.
18. Repeated lexical scaffold explains the common direction — open and important given cosine near1.
19. A general answer-emission direction transfers across unrelated factual relations — untested.
20. A fixed one-dimensional intervention can predict emission changes in unseen scaffold/task combinations — untested; a stronger candidate-unit test than further donor swaps within this scaffold.

Most likely current account: a nearly country-independent, scaffold-dependent interaction component influences whether a direct answer is emitted. This is a useful functional distinction from fact identity, but calling it a fundamental or semantic unit would outrun the evidence. The high cosine may reflect common syntax/position and normalization pathways rather than a learned abstract “answer” variable.

Next distinguish shared wording from shared function. Freeze a direction using only a country subset and one template, then test it on held-out countries, the other capital template where positions align, and different factual relations with matched scaffolds. Keep copy/agreement controls and separate exact-answer rank among a complete declared answer set from generic competition. Include both subtraction and addition, with magnitude matching and native competence reported. Do not tune direction or examples after observing transfer. A mean direction is a proposed intervention, not a unique canonical basis or a model of where the interaction originates.

Audits:48 unique recipient strings,8 unique countries per template, complete56 cross-country pairs per dose, all eight capital candidates included; all source/manifest/basis/runner hashes unchanged; exact native replay; maximum norm mismatch6.38e-7; finite metrics;1,364 evaluations. No new natural-text documents; FineWeb remains26,112. Broad research goal active and incomplete.
