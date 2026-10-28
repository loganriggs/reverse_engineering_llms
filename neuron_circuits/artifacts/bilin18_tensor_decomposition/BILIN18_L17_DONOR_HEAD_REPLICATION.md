

## H338 — Real donor permutations versus source-moment controls

Same H335 recipient documents and intact-state metric. Compare actual source directions with real source directions from six donor-document shifts1,17,37,61,79,101 at the same head and position. Match recipient edit norm and context dot. Average donor Grams before computing cosines. Bootstrap documents and donor shifts jointly; nine-head simultaneous max-deviation bands. Compare the donor average with H337’s six Gaussian moment-control draws using common recipient resampling and independent control-draw/permutation resampling. This tests robustness on an already selected recipient panel, not fresh confirmation.

|Condition|Pooled cosine|Mean head cosine|Head6 MLP Fisher-energy share|
|---|---:|---:|---:|
|actual|-0.456|-0.472|40.23%|
|draw0|-0.474|-0.441|77.84%|
|draw1|-0.480|-0.451|74.40%|
|draw2|-0.461|-0.437|77.53%|
|draw3|-0.478|-0.435|77.54%|
|draw4|-0.474|-0.442|78.48%|
|draw5|-0.473|-0.445|75.42%|
|draw_mean|-0.473|-0.442|76.93%|

|Head|Donor-mean gap [simultaneous95%]|Minimum / maximum per-permutation gap|
|---:|---:|---:|
|0|0.191 [0.039,0.343]|0.171 / 0.212|
|1|0.039 [-0.113,0.191]|0.024 / 0.068|
|2|0.098 [-0.054,0.250]|0.082 / 0.123|
|3|0.177 [0.025,0.329]|0.145 / 0.202|
|4|0.064 [-0.089,0.216]|0.007 / 0.117|
|5|-0.094 [-0.246,0.058]|-0.113 / -0.057|
|6|-0.123 [-0.275,0.029]|-0.137 / -0.105|
|7|-0.009 [-0.161,0.143]|-0.053 / 0.020|
|8|-0.075 [-0.227,0.077]|-0.119 / -0.045|

Locked gates: heads03_repeat: PASS, head6_dominates: PASS, moment_donor_heads03_equivalent5: FAIL. heads03_repeat requires positive donor−actual gaps in all six permutations and both simultaneous lower bounds>0. head6_dominates requires>=65% MLP Fisher energy in head6 in all permutations. moment_donor_heads03_equivalent5 requires moment−donor simultaneous intervals entirely within±0.05 for both heads. Positive gaps mean weaker opposition in the control.

**Moment-control comparison.**

|Head|Moment minus donor cosine [nine-head simultaneous95%]|
|---:|---:|
|0|0.009 [-0.077,0.094]|
|1|0.036 [-0.049,0.122]|
|2|-0.001 [-0.087,0.084]|
|3|0.032 [-0.053,0.118]|
|4|-0.014 [-0.100,0.072]|
|5|0.174 [0.088,0.259]|
|6|0.034 [-0.052,0.119]|
|7|0.038 [-0.047,0.124]|
|8|0.063 [-0.023,0.149]|

Failure of equivalence does not establish a difference. Donor directions retain higher-order distributional structure but destroy recipient pairing; geometry matching itself changes them.

**Limits.** Conditional on checkpoint, recipient panel, donor permutations and intervention construction. Hierarchical resampling estimates recipient-document/control variability but not uncertainty from fitting source moments or selecting these hypotheses. Fisher geometry is a local prediction-sensitive measure, not semantic function. Pooled similarity cannot establish headwise equivalence. Native and synthetic source directions can differ in higher moments and context pairing.

**Audit.** Donor1 and actual Grams replay H335 within1.62e-06 relative; matching error7.63e-07. Small subtraction-order differences permitted. Hashes checked; per-permutation/head/document Grams and bootstrap gaps saved. Native weights unchanged.

**Next discriminator.** If differences survive, distinguish higher-order source distribution from source–context pairing with a head-specific donor-versus-moment comparison, then fresh targeted confirmation. Keep head6 redistribution separate from head0/3 response differences. Broad research goal active.
