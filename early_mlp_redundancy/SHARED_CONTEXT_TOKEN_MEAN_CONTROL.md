# M52: token-conditioned directions explain much of the donor transfer

**The token-conditioned mean beats the same-token context donor on natural text in all six pooled grouping/component comparisons.** Learned shared-component error is 0.2452 for token means, 0.3772 for donors, and 0.6534 for unconditional group means. Zero prediction scores 1. This weakens an interpretation of donor transfer as evidence for context-specific reusable computation. It supports a token-conditioned mean direction as a useful geometric description of these frozen featurizers.

## What was fixed

Use the same eight M38 models, 32 groups, three components, and original M50 eligibility. For each token/group/component, average masked coefficients over all occurrences of that token in natural training documents 0–31, including inactive gates as zero. Decode once with the original normalized dictionary. No fitting or evaluation-based tuning. All predictors receive oracle recipient magnitude, including zero-target information. Unseen tokens use the unconditional group mean; seen tokens with a zero component mean remain zero predictions. This is not a deployable predictor of magnitude or presence.

Coverage is independent of activations and outcomes:

| Cohort | Eligible unique prefixes | Unseen token | One training occurrence | At least two |
|---|---:|---:|---:|---:|
| Task train | 405 | 103 | 12 | 290 |
| Task evaluation | 86 | 18 | 2 | 66 |
| Natural evaluation | 1999 | 365 | 90 | 1544 |

No sparse or unseen examples are removed. Counts are training occurrences, not independent documents or contexts. Existing task holdouts remain unused for fitting; this is still a familiar development evaluation, not new-task discovery.

## Local direction results

Pool error and target energy across both seeds and both source objectives. Different models define different target components; comparisons across architectures are not matched-target claims.

| Cohort | Grouping | Component | Donor | Group mean | Token mean |
|---|---|---|---:|---:|---:|
| task/train | random | exclusive | 0.6586 | 1.5176 | 0.8454 |
| task/train | random | shared | 0.2575 | 1.2929 | 0.3456 |
| task/train | random | union | 0.2777 | 1.3931 | 0.4718 |
| task/train | coactivation | exclusive | 0.4434 | 1.2920 | 0.6694 |
| task/train | coactivation | shared | 0.1756 | 0.7754 | 0.2306 |
| task/train | coactivation | union | 0.1961 | 0.9383 | 0.3190 |
| task/evaluation | random | exclusive | 0.7055 | 2.0374 | 0.5532 |
| task/evaluation | random | shared | 0.3822 | 0.9155 | 0.1278 |
| task/evaluation | random | union | 0.3202 | 1.2015 | 0.2383 |
| task/evaluation | coactivation | exclusive | 0.4937 | 1.0764 | 0.7038 |
| task/evaluation | coactivation | shared | 0.1980 | 0.6790 | 0.2322 |
| task/evaluation | coactivation | union | 0.1989 | 0.7433 | 0.2260 |
| natural/evaluation | random | exclusive | 0.8436 | 1.3950 | 0.5973 |
| natural/evaluation | random | shared | 0.5322 | 1.0713 | 0.3864 |
| natural/evaluation | random | union | 0.5619 | 1.2536 | 0.3792 |
| natural/evaluation | coactivation | exclusive | 0.8144 | 1.1999 | 0.7183 |
| natural/evaluation | coactivation | shared | 0.3772 | 0.6534 | 0.2452 |
| natural/evaluation | coactivation | union | 0.4277 | 0.7494 | 0.2973 |

For natural learned shared components restricted descriptively to tokens with at least two training occurrences, token-mean error is 0.1640 versus donor 0.3699 and group mean 0.5923. This subset holds about 91% of pooled target energy. For unseen tokens, token mean equals group mean by construction (1.1312), and donor is better (0.4177). For singleton tokens, token mean is 0.6011 versus donor 0.6202. These differences show why sparse coverage must stay visible.

## Conceptual interpretation

A stable token-conditioned direction can be spread across multiple atoms and groups. Thus **distributed implementation and token-conditioned function geometry are compatible**, rather than competing labels. Shared architectural atoms can encode this structure without proving duplicated native computation or literal removal redundancy. The context donor may benefit from token identity, while also introducing nuisance context variation that an average suppresses.

The result does not prove context is irrelevant: token means leave nonzero error; an average can outperform one random context even when contextual variation matters causally. Nor does it establish that these featurizers recover the model's fundamental computational units. The task evaluation comparisons are mixed, and token means trained on natural text may face domain shift. Signed means are only one fixed baseline, not a complete token-only optimum.

## Verification and next discriminating experiment

1152 rows include all/sparse-coverage subsets. All 72 donor rows replay M50; a separate audit replays all 216 prior M51 donor/group-mean/scramble rows. Every token coefficient table (768 group/component/model combinations) is independently aggregated with NumPy in the producer. Separate saved-table audit checks coefficient additivity, independent token counts, all coverage partitions, unseen fallback equality, unchanged scramble, and weighted recovery of M51 group means. Maximum independently normalized decoder discrepancy is 5.16e-7, within 1e-6 tolerance.

Next run the frozen native-response comparison in `validation/shared_context_native_protocol.json`. Native response fidelity is not measured by this report. Stop further local direction baseline tuning; compare native effects once and use the result to decide whether shared featurizers expose behaviorally relevant reusable structure or mainly a useful encoding of token-conditioned directions.
