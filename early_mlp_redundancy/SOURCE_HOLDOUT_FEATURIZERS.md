# M59: causal supervision transfers unevenly across positions

**Added causal supervision improves source-position equal-family fidelity in all6 architecture/seed comparisons, but worsens final-position equal-family fidelity in all6.** This qualifies M58's training-panel result: the supervision is not simply failing to transfer, but its benefits do not cover every intervention region or metric.

## Fixed development holdout scope

All14 pre-existing source-group evaluation pairs (28endpoints) are retained, covering entity_binding, capital_recall, antonym, plural_noun, past_tense and country_language. There are no source-group evaluation pairs for agreement, sentiment, pronoun_gender, number_successor or object_color. Those missing families are not filled with training examples. Existing whole-task exclusions remain unused.

These prompts were examined previously in the project. This is source-group/lexical **development transfer**, not pristine whole-task validation or task discovery. Family composition differs from M58: directly comparing the full training pooled score against this holdout pooled score would confound transfer with task coverage.

Exactly84 native MLP1 assignments (changed input positions, final position, identity) are applied identically to all18 frozen M37 states. No fits or target/candidate selection. Both baselines and intervention targets replay exactly. As in M58, external native deltas are added to a reconstructed intact context; the test does not establish that an encoder discovers semantic groups.

## Every final state

Zero response scores1. Pooled error weights by native response energy; the family mean equally weights six family ratios. Small denominator sensitivity remains, so raw energies and per-family values are retained.

| Representation | Seed | Objective | Source pooled | Source family mean | Final pooled | Final family mean |
|---|---:|---|---:|---:|---:|---:|
| scalar | 71 | KL | 0.641 | 1.316 | 0.633 | 0.844 |
| scalar | 71 | KL+response | 0.635 | 1.289 | 0.575 | 1.020 |
| block | 71 | KL | 0.721 | 1.862 | 0.601 | 1.014 |
| block | 71 | KL+response | 0.723 | 1.580 | 0.733 | 1.445 |
| shared_fixed | 71 | KL | 1.007 | 1.679 | 2.007 | 1.711 |
| shared_fixed | 71 | KL+response | 0.755 | 1.332 | 1.000 | 1.733 |
| scalar | 72 | KL | 0.607 | 1.212 | 0.910 | 0.938 |
| scalar | 72 | KL+response | 0.533 | 0.961 | 0.763 | 1.023 |
| block | 72 | KL | 1.016 | 2.379 | 1.156 | 1.233 |
| block | 72 | KL+response | 0.644 | 1.118 | 0.812 | 1.339 |
| shared_fixed | 72 | KL | 1.126 | 1.635 | 0.867 | 1.370 |
| shared_fixed | 72 | KL+response | 0.869 | 1.616 | 1.023 | 1.409 |

Five of six source pooled comparisons improve (block71 is nearly unchanged/slightly worse), while all six source family means improve. Only scalar72 joint reaches a source family mean below1 (0.961). Thus relative improvement does not imply broadly adequate fidelity. Final-position pooled comparisons are mixed, while all six final family means worsen. No architecture/seed is selected from these results.

## Historical objective audit: count balance is not loss-scale balance

The original M35/M36 calibration was training-only:720 scheduled endpoint evaluations across six initial models,72 per family. Each endpoint averages two generator response losses; the training loss uses raw native-probability-weighted squared response error plus intact KL, with a single global response weight7.70262 set by pooled initial K/R. It does not normalize each family by native response energy.

Independently reaggregating calibration records shows capital_recall contributes43.57% of initial response loss, antonym26.80%, and past_tense14.97%. Agreement contributes1.18%, sentiment5.63%. Consequently, **M58's agreement/sentiment-dominated evaluation energy is not the same imbalance as the historical training loss**. Scalar loss contributions are not gradient magnitudes and do not prove the cause of the transfer tradeoff. No causal claim about normalization is established yet.

This identifies a testable hypothesis: equal sampling can leave response objectives dominated by high-scale cells, and a single response objective can improve one intervention region while worsening another. A prospective matched comparison is needed to test whether scale balancing improves broad fidelity, rather than merely changing which headline looks favorable.

## Verification

Native inventory:84forwards,84conditions, all saved vectors/deltas and18summaries independently audited. Candidate comparison:1596forwards,1512response records,504baselines,378summaries,54macros. Every saved final vector and common-delta hash checked with NumPy; maximum candidate metric discrepancy2.22e-15. Native84targets/28baselines replay exactly. Calibration audit verifies all720 initial records, equal family counts, two-generator averaging, pooled K/R and saved loss-source hash. No transformer reimplementation or training-dynamics attribution is claimed.

## Next experiment, not an evaluation-driven weight sweep

Close this observational comparison series. Before any new fit, freeze one three-arm training design from the same six KL-only scalar/block/shared starts: intact KL only; intact KL plus raw common-target response error; intact KL plus training-only family/region-scaled response error. All arms must use identical training examples, schedule, target deltas, update budget and natural-text retention data. Calculate all scales and any fixed stabilizing floor only from training targets before optimization, and use a pooled initial calibration to make objective magnitudes comparable. Do not choose loss settings from these holdout scores.

The concrete next deliverable is that complete prospective protocol plus loss-interface/gradient checks and training-only calibration—not an unreviewed fit. Preserve both source and final targets, identities, weak signals, raw-error reporting and all family holdouts. Reused development evaluation must remain labeled as such. Any claim of improved computational units still requires later manifold/reuse validation and genuine task discovery; loss normalization alone cannot establish either.
