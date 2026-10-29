# Coverage audit and crossed sentiment composition screen

M115 returns the investigation to competent computations and causal targets. The current70-pair causal-training bank covers11named families, but its16sentiment pairs vary adjective polarity without independently varying negation. Historical broader studies already tested negated sentiment, so this is missing factorial coverage in the current bank, not discovery of a previously unknown capability.

## What the inventory ruled out

The older broad competence screen covered17families, and a later expansion covered20candidates. Repeating a generic list of sentiment, arithmetic, retrieval and morphology tasks would not add20new computational domains. The current held-out suite has7explicit operation labels, with important overlaps and competence limits. The M106/M107 retrieval challenges also show why apparent one-key copying cannot be assumed to implement general key selection.

The useful gap chosen here is an independently crossed modifier/valence test. It separates lexical sentiment from a compositional response and permits source-context and output-format comparisons later. It does not substitute one more sentiment template for broad model-wide coverage. Natural-text sampling remains the denominator for any model-wide understanding claim.

## Fixed panel and exclusion policy

Four adjective pairs inherited from training: wonderful/terrible, excellent/awful, great/bad, pleasant/unpleasant. Four subjects: movie, book, meal, game. Two modifiers: really/not. Two formats:

- `Review: The movie was really wonderful. Sentiment:`
- `Text: The movie was really wonderful. Overall sentiment:`

Crossing these factors gives128prompts and128single-factor contrast edges. All edges have equal token lengths; changed positions are recorded. Thirty-two original plain-sentiment endpoints serve as exact replay controls, for160forwards total. The factorial prompts have zero exact overlap with the current training prompts.

The entire new panel stays excluded from featurizer fitting. Positive/negative answer-token groups are inherited unchanged from the current training bank. The diagnostic convention treats really as preserving polarity and not as reversing it; that is a binary sentiment operationalization, not a universal semantics of linguistic negation or a claim that “not excellent” and “terrible” are equivalent.

Each format must pass every valence×modifier cell: at least75% semantic-group-choice accuracy and50% full-vocabulary top1 inside the intended frozen group. For each factor and setting of the other factor, at least75% of contrast edges must have both endpoints correct. Gates were frozen before outputs; there is no prompt repair or answer-group expansion.

## Native results

| Format | Adjective polarity | Modifier | Group correct /16 | Frozen-group top1 /16 |
|---|---|---|---:|---:|
| review | positive | really | 16 | 2 |
| review | positive | not | 16 | 12 |
| review | negative | really | 16 | 13 |
| review | negative | not | 16 | 14 |
| text | positive | really | 16 | 0 |
| text | positive | not | 0 | 0 |
| text | negative | really | 0 | 0 |
| text | negative | not | 16 | 0 |

**Both formats fail the prespecified full gate.** Review is perfect on the two-group contrast:64/64correct, and every64factor edge flips preference in the expected direction with both endpoints correct. It fails because only2/16really-positive examples place a frozen positive-group token first. Overall Review top1 membership is41/64, which would conceal this cell failure if reported alone.

Text chooses the positive group for all64examples. It therefore scores32/64on the balanced binary labels, has0/64correct top1-group memberships, and flips its group choice on0/64factor edges. Historical plain-sentiment controls replay exactly, with32/32group choices and23/32top1-group memberships correct.

## Output form versus compositional response

Inspecting fixed outputs after the result, Review/really-positive top tokens are rating-like `5` (7cases), `4` (3), `Excellent` (4), and `Good` (2). The frozen group includes the latter but misses capitalized Excellent and the ratings. Text often begins with `I`, `Very`, `The`, or a newline. These observations expose limitations of the narrow next-token label convention. They do not retroactively pass the panel or establish full-sentence classification correctness; no answer group was changed.

A post-result factorial decomposition of the positive-minus-negative group logsumexp margin adds useful information. For adjective sign v and modifier sign n (positive/really encoded+1), write the four margins as offset + alpha*v + beta*n + gamma*v*n. This is an exact descriptive decomposition of the selected output score, not localization of a neural circuit.

| Format | Mean offset | Valence term | Modifier term | Interaction term | Positive interaction, cubes /16 |
|---|---:|---:|---:|---:|---:|
| review | 0.153 | 0.378 | 0.226 | 1.311 | 16 |
| text | 1.687 | 0.403 | 0.373 | 0.990 | 16 |

Both formats retain a positive interaction in every lexical/subject cube. Text’s much larger positive offset keeps its binary choice positive despite that interaction. Thus a zero-flip accuracy result does not imply absence of a compositional signal in this output contrast. Conversely, a positive interaction does not prove a general negation mechanism, correct unrestricted generation, or early-MLP implementation. Group logsumexp is itself a nonlinear measurement.

## Consequence for supervision and native-computation hypotheses

No new semantic-supervision cohort is admitted by this screen. Review supplies a clean, observed compositional contrast, while the two formats reveal a distinction between contrast modulation and output-label calibration. Native response vectors can be investigated as descriptive targets without pretending failed intended labels are model behavior. Any subsequent fitting remains subject to the declared exclusions and competence criteria.

The most useful next mechanistic question is whether the same lexical-change signal is reused under different modifier contexts, with the downstream reader changing its interpretation, or whether early MLP writes themselves encode a modifier-dependent computation. A second question is whether format changes mostly alter output offset or also change the causal route. Neither follows from the margin decomposition alone.

Before running this, inspect the prior context-conditioned reader and native-response transfer studies (including M32) to avoid duplication. If a discriminator remains missing, freeze one bounded native patch comparison on the complete panel: both modifiers and formats, direct lexical-change patches versus cross-context transferred changes, source/final sites, native teachers and identity controls, with full distributions and interaction effects retained. Keep failures and do not relabel the formats as competent. No such causal run or feature fit has been started.

## Validation

160native forwards,32bit-exact historical logit replays, all128factor edges aligned. Independent NumPy scoring audits all160saved logits; all tokens, factor labels and edge changes are checked. Maximum margin discrepancy1.78e-15. The descriptive factorial coefficients reconstruct every corner and agree with independent signed averaging. All jobs are terminal.

Scripts and receipts: build_sentiment_composition.py, probe_sentiment_composition.py, audit_sentiment_composition.py, describe_sentiment_composition.py, and validation/sentiment_composition_*.json. Native logits and MLP0–2normalized inputs/product writes are saved locally at the path and hash in the native receipt. The protocol records historical screen and training-inventory hashes.
