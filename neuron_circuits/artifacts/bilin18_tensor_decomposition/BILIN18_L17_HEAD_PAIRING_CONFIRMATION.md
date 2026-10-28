

## H339 — Fresh head-specific pairing confirmation

New128 recipient documents exclude4160 complete-document hashes from prior panels; source documents/windows independently audited. Frozen H333 means/covariance, six new Gaussian draws, six fixed donor shifts1,17,37,61,79,101, recipient norm/context-dot matching and a common intact-state Fisher metric. No fitting or tuning on recipients. Test three planned contrasts across all9 heads: donor−actual, moment−actual and moment−donor. Joint4000-fold document/control bootstrap; one max-absolute-deviation95% band covers all27 contrasts. This is experimental-panel freshness, not a model-training holdout.

|Condition|Pooled cosine|Head6 MLP Fisher-energy share|
|---|---:|---:|
|actual|-0.407|39.79%|
|moment|-0.412|76.29%|
|donor|-0.436|76.88%|

|Contrast|Head|Gap [joint27 simultaneous95%]|
|---|---:|---:|
|donor_actual|0|0.199 [-0.038,0.436]|
|donor_actual|1|0.045 [-0.192,0.282]|
|donor_actual|2|0.067 [-0.169,0.304]|
|donor_actual|3|0.089 [-0.148,0.326]|
|donor_actual|4|0.050 [-0.187,0.286]|
|donor_actual|5|-0.320 [-0.556,-0.083]|
|donor_actual|6|-0.148 [-0.385,0.088]|
|donor_actual|7|-0.028 [-0.265,0.209]|
|donor_actual|8|-0.084 [-0.321,0.153]|
|moment_actual|0|0.220 [-0.017,0.457]|
|moment_actual|1|0.079 [-0.158,0.316]|
|moment_actual|2|0.084 [-0.152,0.321]|
|moment_actual|3|0.082 [-0.154,0.319]|
|moment_actual|4|0.007 [-0.230,0.243]|
|moment_actual|5|-0.102 [-0.339,0.135]|
|moment_actual|6|-0.118 [-0.355,0.119]|
|moment_actual|7|0.010 [-0.227,0.247]|
|moment_actual|8|-0.046 [-0.283,0.190]|
|moment_donor|0|0.021 [-0.216,0.258]|
|moment_donor|1|0.034 [-0.203,0.271]|
|moment_donor|2|0.017 [-0.220,0.254]|
|moment_donor|3|-0.006 [-0.243,0.230]|
|moment_donor|4|-0.043 [-0.280,0.194]|
|moment_donor|5|0.218 [-0.019,0.454]|
|moment_donor|6|0.031 [-0.206,0.267]|
|moment_donor|7|0.038 [-0.199,0.275]|
|moment_donor|8|0.038 [-0.199,0.274]|

Locked gates: heads03_pairing: FAIL, head5_distribution: FAIL, head6_concentration: PASS. heads03_pairing requires donor−actual gaps>=0.1 and simultaneous lower bounds>0 at both heads0/3. head5_distribution requires moment−donor gap>=0.1 and lower>0 at head5. head6_concentration requires>=65% MLP Fisher energy at head6 in both control averages.

**Interpretation limits.** Donor directions preserve real source distribution but geometry matching transforms them; therefore donor mismatch is consistent with pairing dependence, not uniquely semantic binding. Moment–donor differences can reflect higher-order statistics and other distribution mismatches. Fisher-energy dominance is not semantic importance. Conditional resampling includes recipient/control variability but not source-moment fitting or checkpoint uncertainty; donor permutation dependencies are not independent fresh data. The previous explorations selected these hypotheses, so only this frozen targeted test is confirmation.

**Audit.** Initial summary serialization failed on NumPy booleans after all per-document Grams had been saved. The error log was preserved; boolean serialization and a duplicate cached-bootstrap-key recovery error were corrected (both logs preserved), and the summary was recovered from cached Grams with a repeated geometry audit, without new model-response outcomes. Maximum norm/dot matching error6.97e-07; frozen geometry/moment and corpus hashes checked. Document/window provenance audit passed. Saved all117 head/condition per-document Grams and joint bootstrap differences; native weights unchanged.

**Next discriminator.** Use confirmed head-specific effects to localize which compact compiled modes or remaining MLP paths carry the source–context dependence. Require finite behavioral intervention controls before semantic labeling. Preserve failed confirmation gates as limits on the proposed ontology. Broad research goal active.
