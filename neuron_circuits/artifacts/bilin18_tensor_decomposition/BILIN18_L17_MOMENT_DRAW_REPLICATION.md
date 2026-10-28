

## H337 — Independent draws of the source-moment control

Same H335 recipient documents, frozen H333 source moments and common intact-state Fisher metric. Replay original Gaussian seed and add five independent seeds. Compute actual and all six mean-plus-covariance controls, matching recipient norm/context dot. Average Grams across draws before computing the principal control cosine. Jointly bootstrap recipient documents and global draw indices4000 times, preserving paired heads; construct max-absolute-deviation95% bands across all9 heads. Heads0/3 were selected by H336, so this is control-draw robustness, not fresh-document confirmation. Six draws provide limited evidence about the full draw distribution.

|Condition|Pooled cosine|Mean head cosine|Head6 MLP Fisher-energy share|
|---|---:|---:|---:|
|actual|-0.456|-0.472|40.23%|
|draw0|-0.467|-0.405|76.16%|
|draw1|-0.420|-0.403|72.15%|
|draw2|-0.438|-0.394|74.39%|
|draw3|-0.464|-0.401|73.92%|
|draw4|-0.465|-0.408|75.06%|
|draw5|-0.429|-0.393|73.53%|
|draw_mean|-0.447|-0.401|74.26%|

|Head|Draw-mean gap [simultaneous95%]|Minimum / maximum per-draw gap|
|---:|---:|---:|
|0|0.200 [0.044,0.355]|0.177 / 0.228|
|1|0.076 [-0.080,0.232]|0.053 / 0.117|
|2|0.096 [-0.059,0.252]|0.069 / 0.132|
|3|0.209 [0.053,0.365]|0.173 / 0.233|
|4|0.049 [-0.106,0.205]|0.018 / 0.116|
|5|0.080 [-0.076,0.235]|0.041 / 0.113|
|6|-0.090 [-0.245,0.066]|-0.124 / -0.044|
|7|0.029 [-0.127,0.185]|-0.023 / 0.070|
|8|-0.012 [-0.168,0.144]|-0.047 / 0.049|

Locked gates: heads03_repeat: PASS, head6_dominates: PASS. heads03_repeat requires both simultaneous lower bounds>0 and positive gaps in every draw. head6_dominates requires>=65% of MLP Fisher energy in head6 for every draw. Positive gaps mean weaker opposition in the control.

**Limits.** Conditional on source moment estimates, checkpoint, recipient panel and intervention construction. Hierarchical resampling estimates document/draw variability but not uncertainty from fitting source moments or selecting these hypotheses. Fisher geometry is a local prediction-sensitive measure, not semantic function. Pooled similarity cannot establish headwise equivalence. Native and synthetic source directions can differ in higher moments and context pairing.

**Audit.** Original-seed and actual Grams replay H335 within9.91e-07 relative; matching error8.11e-07. Small subtraction-order differences permitted. Hashes checked; per-draw/head/document Grams and bootstrap gaps saved. Native weights unchanged.

**Next discriminator.** If differences survive, distinguish higher-order source distribution from source–context pairing with a head-specific donor-versus-moment comparison, then fresh targeted confirmation. Keep head6 redistribution separate from head0/3 response differences. Broad research goal active.
