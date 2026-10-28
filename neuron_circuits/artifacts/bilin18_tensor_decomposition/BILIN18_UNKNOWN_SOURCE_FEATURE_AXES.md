# Unknown source-feature axes beyond the human vocabulary

Human category centroids are subtracted from each token's six-layer routing profile. Weighted residual axes are fit on one FineWeb split, aligned to an independent split, and used prospectively to predict the second split. Each candidate saves 50 positive and 50 negative tokens plus up to 50 corpus contexts per pole.

## Locked hypotheses

- **U1_human_categories_leave_structure: not supported.**
- **U2_residual_axes_beat_random: not supported.**
- **U3_multiple_residual_axes_are_stable: supported.**
- **U4_unknown_axes_exist: supported.**
- **U5_residual_recovery_scales_with_rank: not supported.**

Residual recovery by rank: 4: -0.100, 8: -0.064, 16: -0.288, 32: -0.411, 64: -0.411.

## Strongest candidate unknown axes

- **U5**: split axis cosine 0.952, token-score correlation 0.830, closest known `determiner` (-0.402). Positive: ' class', ' US', ' He', ' function', ' we', ' us', 'am', ' She', ' him', ' Be'. Negative: ' perhaps', ' Other', ' mailing', ' lovely', ' respectable', ' Instead', 'Another', ' remaining', 'Later', ' stupid'. Functions: L0 H6 d[4,8), L0 H6 d[2,4), L0 H5 d[0,1), L0 H6 d[0,1), L2 H2 d[1,2).
- **U4**: split axis cosine 0.962, token-score correlation 0.786, closest known `determiner` (-0.291). Positive: ' Instead', 'Either', 'Later', 'Sign', '1977', ' Unlike', ' Gone', ' Claim', '700', 'Remove'. Negative: 'all', 'he', ' flexibility', 'IS', ' oste', 'an', ' Nations', 'oph', 'ru', ' Champions'. Functions: L3 H0 d[8,16), L1 H5 d[1,2), L0 H4 d[1,2), L3 H7 d[1,2), L0 H4 d[4,8).
- **U10**: split axis cosine 0.970, token-score correlation 0.747, closest known `conjunction` (+0.473). Positive: ' Pearl', ' wildfire', ' Elements', ' slim', ' criminal', ' disobedience', ' canal', ' irrelevant', 'iny', ' Dame'. Negative: ' >>>', ' ill', ' diagnosis', 'Over', 'Non', 'over', 'umbo', ' 109', ' rental', 'nothing'. Functions: L0 H0 d[2,4), L0 H0 d[4,8), L0 H0 d[8,16), L0 H0 d[1,2), L5 H5 d[2,4).
- **U7**: split axis cosine 0.954, token-score correlation 0.737, closest known `determiner` (-0.348). Positive: 'ru', 'uc', ' (.', 'anc', 'aked', 'ut', 'em', 'all', 'ct', 'ign'. Negative: ' Journal', ' USDA', ' Financial', ' draft', '44', ' oil', ' Baby', ' poverty', ' SEO', '551'. Functions: L0 H4 d[1,2), L0 H6 d[1,2), L0 H4 d[8,16), L0 H4 d[4,8), L0 H4 d[16,32).

## Guardrails

These are token-level routing-role axes, not yet semantic concepts. The source profile uses magnitude and discards sign. Prediction transfers across two FineWeb halves but uses the same token identities, so it tests role stability rather than unseen-token generalization. Existing-feature correlations cover the current 50-label audit only. A low correlation means unknown to that vocabulary, not necessarily novel to a human. Context examples are discovery aids and are not used in the locked quantitative gates.
