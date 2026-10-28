

## H336 — Head and path heterogeneity behind pooled cosine

Exploratory stratification of saved H334/H335 per-document Fisher Grams; both pooled panels were already observed. No new model runs or fresh confirmation claim for this analysis. For each head, compute direct–MLP, direct–compiled and direct–remaining cosines. Compare mean_cov, centered and donor controls to actual, with4000 joint paired document bootstrap draws. Construct a common max-absolute-deviation95% band across nine heads separately for each panel/contrast/path. These bands do not correct across every contrast and path.

|Panel|Head|Actual direct–MLP|Moment direct–MLP|Gap [nine-head simultaneous95%]|
|---|---:|---:|---:|---:|
|discovery|0|-0.385|-0.221|0.164 [-0.023,0.350]|
|discovery|1|-0.477|-0.381|0.096 [-0.091,0.282]|
|discovery|2|-0.435|-0.323|0.112 [-0.074,0.299]|
|discovery|3|-0.681|-0.544|0.137 [-0.050,0.323]|
|discovery|4|-0.643|-0.650|-0.007 [-0.194,0.179]|
|discovery|5|-0.403|-0.380|0.023 [-0.164,0.209]|
|discovery|6|-0.426|-0.535|-0.108 [-0.295,0.078]|
|discovery|7|-0.220|-0.277|-0.057 [-0.244,0.129]|
|discovery|8|-0.562|-0.526|0.036 [-0.151,0.222]|
|confirmation|0|-0.436|-0.240|0.196 [0.036,0.356]|
|confirmation|1|-0.486|-0.423|0.064 [-0.096,0.223]|
|confirmation|2|-0.425|-0.292|0.132 [-0.028,0.292]|
|confirmation|3|-0.664|-0.449|0.215 [0.055,0.375]|
|confirmation|4|-0.682|-0.628|0.054 [-0.106,0.214]|
|confirmation|5|-0.347|-0.265|0.082 [-0.078,0.242]|
|confirmation|6|-0.408|-0.526|-0.118 [-0.277,0.042]|
|confirmation|7|-0.295|-0.268|0.027 [-0.133,0.187]|
|confirmation|8|-0.505|-0.552|-0.047 [-0.207,0.112]|

**Finding.** Confirmation-panel heads0 and3 have gaps+0.196/+0.215 with simultaneous bands excluding zero: the control opposes less than actual there. Head6 shifts the other way, and its share of MLP Fisher energy rises40.2%→76.2%. Equal-head mean cosine changes−0.472→−0.405, while pooled cosine changes−0.456→−0.467. Therefore pooled closeness can conceal redistribution across heads. Path-specific differences also remain: on the confirmation panel, direct–compiled bands exclude zero at heads0/6 and direct–remaining at heads3/6. These are separately corrected families, not one global family.

Across panels, head-gap correlation=0.853; same-sign heads=6/9. Locked exploratory gates: fresh_head_heterogeneity: PASS, head_pattern_repeats: FAIL. Heterogeneity requires max gap>=0.15 and at least one simultaneous band excluding zero; pattern repetition requires correlation>=0.7 and>=7 same signs. The latter fails despite a high correlation.

**Interpretation limits.** A Fisher-energy share is a sensitivity-weighted magnitude, not a semantic importance score. Head6 dominance changes weighting in pooled statistics; it does not by itself explain all gaps. Bootstrap conditions on frozen models, source statistics and a single Gaussian draw. Head selection is exploratory; future targeted tests must be separately confirmed.

**Audit.** Pooled cosines recomputed from saved rows reproduce original summaries to2.22e-16. Input row-file hashes checked. Saved all per-head/path metrics and bootstrap gap arrays. Native weights unchanged.

**Next discriminator.** Test source-moment controls with head-balanced metrics and several independent draws. Then target head0/3 pairing-sensitive effects and head6 redistribution, keeping compiled and remaining paths separate. Broad research goal remains active.
