# M51: conditional donors versus training-only group directions

The same-token, same-group donor has lower pooled local direction error than an unconditional group mean in every cohort/grouping/component comparison. For natural learned shared components, donor error is 0.3772 versus group-mean 0.6534. Zero prediction scores 1. This excludes a single average direction per group as the full explanation, but does **not** establish context-specific computation: token identity alone may explain the extra predictability.

## Fixed comparison

All eight frozen M38 models and the exact M50 eligible unique-prefix population are retained. Group means use only natural training documents 0–31 (4096 positions), including zero contributions when the group/component is inactive. Means are computed in coefficient space and decoded with the original normalized dictionary. No regression, architecture, strength, seed, or membership search. Each predictor receives oracle recipient magnitude and zero-target information. Missing donor or mean directions remain zero. The scrambled control uses one fixed coordinate permutation, seed 51001, shared by all models and examples. It preserves magnitude but generally leaves the learned decoder subspace; it is a coarse negative control, not a matched semantic null.

Errors below pool squared errors and target energies across two seeds and two source objectives. Different candidates have different target interventions; this is not a matched-target architecture ranking. Task cohorts are familiar development panels, not fresh discovery tests.

| Cohort | Groups | Component | Donor | Group mean | Scrambled |
|---|---|---|---:|---:|---:|
| task/train | random | exclusive | 0.6586 | 1.5176 | 2.0170 |
| task/train | random | shared | 0.2575 | 1.2929 | 1.9989 |
| task/train | random | union | 0.2777 | 1.3931 | 2.0037 |
| task/train | coactivation | exclusive | 0.4434 | 1.2920 | 2.0264 |
| task/train | coactivation | shared | 0.1756 | 0.7754 | 2.0398 |
| task/train | coactivation | union | 0.1961 | 0.9383 | 2.0399 |
| task/evaluation | random | exclusive | 0.7055 | 2.0374 | 1.9912 |
| task/evaluation | random | shared | 0.3822 | 0.9155 | 1.9689 |
| task/evaluation | random | union | 0.3202 | 1.2015 | 1.9887 |
| task/evaluation | coactivation | exclusive | 0.4937 | 1.0764 | 2.0490 |
| task/evaluation | coactivation | shared | 0.1980 | 0.6790 | 2.0500 |
| task/evaluation | coactivation | union | 0.1989 | 0.7433 | 2.0501 |
| natural/evaluation | random | exclusive | 0.8436 | 1.3950 | 2.0019 |
| natural/evaluation | random | shared | 0.5322 | 1.0713 | 2.0141 |
| natural/evaluation | random | union | 0.5619 | 1.2536 | 2.0082 |
| natural/evaluation | coactivation | exclusive | 0.8144 | 1.1999 | 2.0042 |
| natural/evaluation | coactivation | shared | 0.3772 | 0.6534 | 2.0237 |
| natural/evaluation | coactivation | union | 0.4277 | 0.7494 | 2.0182 |

## Verification and limits

All 72 donor summary rows reproduce M50 within 1e-12 relative tolerance. Independent NumPy reconstruction of all 32 groups × 3 components × 8 models uses all training positions; maximum relative mean discrepancy is 1.25e-16. Exclusive plus shared means equal union means. Training/evaluation document separation and source hashes are checked. There are 216 result rows. No new native forward passes were needed, and native response fidelity is still unmeasured here.

The group-mean baseline is unconditional on token; the donor is conditioned on token. Therefore the comparison cannot identify context-specific information beyond token identity. Signed means may also be weak summaries of multimodal directions. A stronger next control is a training-only token/group/component direction, with sparse or unseen-token coverage explicitly reported and unconditional fallback. Do not silently select only well-supported tokens. No outcome-based donor or seed selection. After freezing this control, compare native response prediction against zero, unconditional mean, token-conditioned mean, donor, and the fixed scramble, on the original eligible population. Treat this as a bounded comparison and stop architecture tuning. This connects geometric sharing to causal reuse without equating either with literal removable native redundancy.
