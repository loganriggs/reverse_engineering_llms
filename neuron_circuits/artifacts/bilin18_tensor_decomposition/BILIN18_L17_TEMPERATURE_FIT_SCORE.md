

## H380 — Cross-scoring exposes strong fit–metric interaction

Cross H379’s fit temperatures0.5/1/2 with scoring temperatures0.5/1/2, keeping each fit’s rank16/64 coefficients and its selected finite step fixed. No new fitting or step selection occurs when the scoring temperature changes. Full native activation targets and the first frozen random span remain fixed. These are nine combinations of modified output distributions, not nine native models. Reuse the512-document panel.

|Head|Family|Cold-fit/cold-score64|Cold-fit/native-score64|Native-fit/native-score64|Cold-fit/hot-score64|Hot-fit/cold-score64|
|---|---|---:|---:|---:|---:|---:|
|0|actual|0.987|0.721|0.810|-0.075|0.646|
|0|moment|0.984|0.605|0.725|-0.533|0.573|
|0|donor|0.984|0.612|0.730|-0.504|0.578|
|1|actual|0.977|0.681|0.768|-0.169|0.551|
|1|moment|0.977|0.653|0.750|-0.252|0.545|
|1|donor|0.977|0.656|0.753|-0.247|0.538|
|6|actual|0.978|0.685|0.778|0.002|0.502|
|6|moment|0.985|0.757|0.822|0.307|0.257|
|6|donor|0.984|0.761|0.824|0.319|0.227|

Cold-fit rank64 recovery drops from97.7–98.7% under cold scoring to60.5–76.1% at native temperature and−53.3% to31.9% under hot scoring. Hot-fitted64 coefficients score only22.7–64.6% under cold scoring. Thus cold scoring alone does not ensure near-perfect recovery, and a fit optimized for that scoring distribution does not generally transfer. The finite safeguard is guaranteed only at the fit temperature where its step was selected; off-diagonal cells can be worse than baseline.

For the cold/hot2×2 corners, report symmetric factor contributions to R(cold,cold)−R(hot,hot): fitting contribution averages the fit change at both score temperatures; scoring contribution averages the score change at both fit temperatures. Contributions sum exactly to the diagonal contrast. Shares divide by that contrast. The interaction is R(cold,cold)−R(cold,hot)−R(hot,cold)+R(hot,hot).

All frozen gates fail: {'scoring_majority': False, 'fitting_majority': False, 'native_fit_advantage': False}. Scoring shares exceed50% with joint lower bounds for heads0/1 and head6 actual, but not head6 controls. Fitting is likewise not a universal majority. Scoring point shares range34–128%; shares above100% reflect an opposing averaged fitting contribution, not more than100% of literal behavior. Interaction point values span0.915–1.609 recovery units, so averaged attribution should not be treated as a single context-independent mechanism.

At native scoring, native-fit coefficients outperform cold-fit coefficients in every head/family/rank joint lower bound. Rank16 point gains are13.1–21.6pp and rank64 gains6.3–12.0pp. The frozen universal10pp lower-bound requirement fails. This supports the importance of matching fitting geometry to evaluation geometry without claiming the stronger failed margin.

Audits: {'h379_diagonal_replay_relative': 5.721189558844275e-15}. All diagonal cells replay H379 within5.8e-15 relative error; finite KL values pass positivity/finite checks; symmetric attribution recombines exactly within tolerance. Input hashes verified. Save all fit×score site-level KLs, scoring-specific baseline KLs and paired bootstrap arrays. Native weights and response vectors unchanged.

Uncertainty:4,000 paired document/control-draw bootstrap samples; joint36 bands cover both factor-share estimates across18 groups, and separate joint18 bands cover native-fit advantage. Fixed one random orientation, discovery summaries, fitted coefficients and empirical donor assignments. Reused panel. Figure displays unweighted averages of the three family recovery estimates for each head for readability, not a pooled population estimate; tables/JSON preserve each group.

Research update: apparently compact behavioral descriptions can depend heavily on both the metric and context-specific fitting. High matched-metric oracle recovery alone is weak evidence of fixed computational units. The native coefficient-generation problem remains unsolved, and arbitrary directions cannot be dismissed solely from their lower matched-metric score.

Next: at unchanged native scoring, compare recipient-specific oracle coefficients with coefficients fitted under the frozen shared probability metric. Keep the same random/probability spans and finite safeguard. This isolates the value of recipient-specific coefficient adaptation without changing the scored output distribution, before deciding which coefficient-generation approaches merit held-out prediction experiments. Broader goal remains active.
