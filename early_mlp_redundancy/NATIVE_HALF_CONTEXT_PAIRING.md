# M46: correct within-token context pairing carries predictive information

M45 left open whether the retained-half correction helps mainly by correcting a noisy token table. M46 fixes token identity and prediction marginals, then compares correct pairing against the exact expected error of another context's prediction. No new fit, strength, or shuffle seed is selected.

Within each domain/source split, deduplicate exact causal prefixes. For each training-known token with n>=2 distinct evaluation prefixes, let p_i be the frozen predicted missing-half residual and y_i the target residual relative to its training token mean. Compare paired squared error with uniform other-context assignment. Algebra gives:

    wrong-context SSE - paired SSE
      = 2n/(n-1) sum_i (p_i-mean(p)) dot (y_i-mean(y)).

Token-constant offsets cancel. A positive gain therefore reflects within-token prediction–target covariance, not a constant token correction alone. It can still reflect position, topic, shared inputs, or other context variables rather than a semantic operation. The exact off-diagonal average is not a confidence interval.

## Coverage and results

Unknown-token and single-context positions are excluded and counted, not classified as failures. No donors cross domain/source-split boundaries. Natural eligibility covers54.6% of original positions. Task eligibility is limited to common repeated tokens and does not imply coverage of every task computation.

| Cohort | Original positions | Unique prefixes | Eligible prefixes | Eligible token types |
|---|---:|---:|---:|---:|
| task train | 1022 | 578 | 303 | 28 |
| task evaluation | 256 | 140 | 69 | 12 |
| natural evaluation | 4096 | 4095 | 2238 | 250 |

| Cohort | Layer | Paired / wrong-context error | Paired / token-mean error | Positive token groups | Largest token share of gain |
|---|---|---:|---:|---:|---:|
| task train | MLP0 | 0.9068 | 0.9136 | 26/28 | 52.2% |
| task train | MLP1 | 0.9489 | 0.9530 | 28/28 | 34.1% |
| task train | MLP2 | 0.9667 | 0.9653 | 27/28 | 25.5% |
| task evaluation | MLP0 | 0.9356 | 0.9079 | 12/12 | 42.9% |
| task evaluation | MLP1 | 0.9543 | 0.9363 | 12/12 | 40.9% |
| task evaluation | MLP2 | 0.9619 | 0.9504 | 12/12 | 32.2% |
| natural evaluation | MLP0 | 0.7722 | 0.8726 | 247/250 | 12.4% |
| natural evaluation | MLP1 | 0.7727 | 0.8637 | 250/250 | 41.0% |
| natural evaluation | MLP2 | 0.8933 | 0.9446 | 249/250 | 5.8% |

Correct pairing lowers error in all nine layer/cohort aggregates. Natural gains are22.8%,22.7%,10.7% atMLP0/1/2. MLP1 natural pairing gains are positive for all250eligible token groups after pooling masks, but one token accounts for41%of total gain. Sign agreement is broad while effect energy is concentrated; contexts and masks are not independent statistical replications.

The native halves thus share some predictable variation beyond token identity. This weakens token-constant correction as a complete explanation. It does not establish functional interchangeability: M42–M43 showed that useful predictions can harm downstream outputs at full strength. Position and common-input covariance remain possible explanations. Reused development data, two dependent half masks and limited coverage constrain generalization.

Next: freeze a deterministic within-token permutation on these eligible prefixes and compare paired versus permuted native missing-half replacements. Preserve all means, maps, cohorts and eligibility; leave native writes untouched at ineligible positions in both arms. This supplies a causal pairing control for featurizers without refitting or optimizing amplitude. No semantic-unit or new-task claim.

## Audit

192native forwards collect4813unique prefixes. Independent audit verifies1740token–mask groups with explicit NumPy all-pairs distance matrices, reconstructs all prefixes/exclusions, checks the covariance identity and every summary, and exactly replays six previously audited native captures. Maximum relative expected-error discrepancy1.15e-15. This stage measures local statistical pairing, not downstream causal consequences.

[Protocol](validation/native_half_pairing_protocol.json) · [Results](validation/native_half_pairing.json) · [Audit](validation/native_half_pairing_audit.json)
