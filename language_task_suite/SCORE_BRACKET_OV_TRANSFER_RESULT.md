# OV reductions must preserve two learned rules together

The [composition pilot](SCORE_BRACKET_COMPOSITION_RESULT.md) gives two behaviorally competent bilinear checkpoints trained on separate score-copy and bracket reports, with no jointly edited reports in training. This method test fits each attention head's exact `O_h V_h` product using either ordinary weight SVD or an activation-covariance-weighted SVD. The covariance comes from 80 score-only and 80 bracket-only train-partition plans, never from joint examples. All heads in both layers are reduced at the same rank. Full rank 16 is an exact balanced gauge: maximum probe-logit changes are 3.4e-5 and 4.0e-5.

On the common 28-game joint heldout panel, native paired bracket/score accuracy is .964/1.000 for seed55801 and .911/.929 for seed55802. At rank 4, neither unsupervised method retains both rules. At rank 8:

| Checkpoint | Reduction | Bracket paired | Score paired | Bracket KL | Score KL | Background KL |
|---|---|---:|---:|---:|---:|---:|
| 55801 | weight | .839 | .714 | .426 | .568 | .433 |
| 55801 | activation-aware | .964 | .536 | .022 | .740 | .316 |
| 55802 | weight | .786 | .750 | .269 | .510 | .329 |
| 55802 | activation-aware | .607 | .732 | .718 | .424 | .238 |

The activation-aware reduction lowers mean input-weighted OV error and background KL, but it does not consistently protect both rules. Its task tradeoff even reverses between seeds. This is a specific failure of average reconstruction as a sufficient rule-preservation objective, not evidence that weight SVD is universally preferable.

To test a rule-aware objective without leaking the joint combination, I initialized rank-8 weight-SVD factors and fitted only their retained OV coordinates to teacher predictions on separate-rule reports, with a 0.1 background-KL term. All other parameters and discarded channels stayed frozen; the maximum absolute discarded-channel value remained exactly zero. I fitted three scopes using the same data panel and initialization: bracket sites only, score sites only, and both.

| Checkpoint | Fit scope | Bracket paired | Score paired | Bracket KL | Score KL |
|---|---|---:|---:|---:|---:|
| 55801 | bracket | .964 | .786 | .031 | .486 |
| 55801 | score | .911 | .929 | .241 | .270 |
| 55801 | both | .964 | .929 | .028 | .271 |
| 55802 | bracket | .929 | .661 | .104 | .722 |
| 55802 | score | .839 | .839 | .234 | .204 |
| 55802 | both | .929 | .875 | .105 | .199 |

Joint rule-site fitting recovers much of both heldout behaviors; fitting only one generally sacrifices the other. The result supports a simple conclusion: at fixed reduced OV capacity, the supervised objective determines which known computations survive. It does **not** show that an unsupervised method discovered reusable structure or that these OV factors are a canonical decomposition of the attention function. The main open test is whether a component selected without a task label can predict a genuinely new rule, plus a stronger holdout over score values, genres, and linguistic realizations.

Reproduce with `eval_score_bracket_ov.py` and `fit_score_bracket_ov_transfer.py`. Aggregate files are `results/score_bracket_ov_seed{55801,55802}_n80_n28.json` and `results/fit_score_bracket_ov_seed{55801,55802}_rank8_{bracket,score,both}.json`. Source passages and model checkpoints are not committed.
