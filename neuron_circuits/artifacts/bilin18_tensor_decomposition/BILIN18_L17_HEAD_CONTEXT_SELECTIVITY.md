

## H358 — Natural response selectivity versus control placement

Reuse H357 exact native output Rayleigh arrays. For actual and each control family separately, define placement gain as geometric own-context Rayleigh divided by geometric confidence-matched-context Rayleigh. Average controls at each site before logs, matching H357. Log pairing equals log control placement minus log actual placement exactly. Placement gain measures local directional probability sensitivity, not prediction quality.

Use4000 joint document/control-realization/neighbor-rank resamples. All-head placement bands cover27 entries jointly. Separately form joint3 bands for head6/head0 placement comparisons. Heads0 and6 were selected after H357: this contrast is exploratory, not independent confirmation.

|Head|Actual placement|Moment placement|Donor placement|
|---|---:|---:|---:|
|0|1.155|1.024|1.034|
|1|0.630|0.694|0.668|
|2|1.154|1.145|1.145|
|3|1.436|1.274|1.307|
|4|0.934|1.008|0.967|
|5|1.380|1.187|1.307|
|6|0.720|0.983|0.940|
|7|1.177|1.173|1.189|
|8|1.344|1.329|1.355|

|Head6/head0 placement|Ratio [joint3 95% band]|
|---|---:|
|actual|0.623 [0.559, 0.695]|
|moment|0.960 [0.861, 1.070]|
|donor|0.909 [0.816, 1.013]|

Both frozen gates pass: actual head6/head0 upper bound<0.8, while both control bands lie entirely inside[0.8,1.25]. Equivalence is only within this declared25% multiplicative margin, not equality. The contrast is concentrated in the natural-response placement rather than a similarly large difference in control placement.

Head1 provides a useful negative comparison: own-context sensitivity is low for actual and both controls (0.630/0.694/0.668). Head6 differs because actual placement is0.720 while controls are0.983/0.940. Thus low natural placement alone is not sufficient for a distinctive control contrast. These numbers are ratios of geometric averages, not a claim that every site behaves similarly.

Absolute placement retains possible context-metric scale differences. Those scale factors are common across heads at each site, so they cancel in the head6/head0 log-placement contrast. Conditional matching geometry, source distributions and semantic roles remain unresolved alternatives.

Audit: the decomposed pairing ratios replay all18 H357 estimates within absolute error 7.77e-15; cached input hash verified. Bootstrap log placements saved. No new model run or native weight change.

Limits: reused data and outcome-selected heads; pointwise/bootstrap family choices are explicit, but fixed neighbor/donor dependencies and selection uncertainty are not fully captured. No semantic unit is identified.

Next discriminator: compare source-specific natural alignment with matched native response directions transferred across contexts while controlling residual direction as well as confidence. Use head1 alongside heads0/6 to distinguish generic low sensitivity from natural-payload coordination; confirm selected-head hypotheses on separate data before broad claims. Broad research goal remains active.
