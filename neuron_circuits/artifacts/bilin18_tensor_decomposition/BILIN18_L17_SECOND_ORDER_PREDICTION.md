

## H392 — Consistent second-order Taylor response prediction

Reuse the H390512-document panel, H391 amplitudes0.5/1/2/4, all39 responses, and the same MLP-only intervention. Predict without the finite target. With u1,u2 the first two Taylor coefficients of RMS(pre−a payload), m1=2B(x,u1), m2=2B(x,u2)+B(u1,u1). The normalized response coefficients are c1=Drms(state,m1) and c2=Drms(state,m2)+0.5 D²rms(state)[m1,m1]. Compare a c1 with a c1+a²c2, followed by exact unembedding, logit cap and softmax. There is no extra RMS after this normalized-displacement prediction, no rank projection, no fitting, and no step safeguard. Native weights and known intervention payload remain required.

The half-second-derivative formula for RMS(v) in direction h is −h mean(vh)/s³ + v[1.5 mean(vh)²/s⁵−0.5 mean(h²)/s³], s²=mean(v²)+epsilon. This includes input and output normalization curvature consistently with the bilinear term, rather than correcting isolated pieces exactly. The finite target is used only for evaluation.

|Head|Family|First1|Second1|First2|Second2|First4|Second4|
|---|---|---:|---:|---:|---:|---:|---:|
|0|actual|0.9925|1.0000|0.9641|0.9999|0.7991|0.9986|
|0|moment|0.9838|1.0000|0.9320|0.9999|0.7338|0.9986|
|0|donor|0.9839|1.0000|0.9322|0.9999|0.7318|0.9986|
|1|actual|0.9887|0.9999|0.9448|0.9976|0.6811|0.9531|
|1|moment|0.9856|0.9998|0.9327|0.9973|0.6641|0.9554|
|1|donor|0.9865|0.9999|0.9360|0.9974|0.6685|0.9545|
|6|actual|0.9672|0.9995|0.8228|0.9919|0.0992|0.8979|
|6|moment|0.9678|0.9989|0.9058|0.9877|0.4108|0.5337|
|6|donor|0.9676|0.9988|0.9056|0.9865|0.4439|0.5457|

Frozen gates: {'second_unit99': True, 'second_two95': False, 'second_four90': False, 'gain_four5': False}. Require all9 second-order lower recovery bounds >99% at1×, >95% at2×, >90% at4×; and all9 lower second-minus-first gains >5pp at4×.

Audits: {'baseline_h391_replay_max_absolute': 8.881784197001252e-16, 'first_order_h391_replay_max_absolute': 1.7763568394002505e-15, 'second_taylor_coefficient_fd_max_relative': 4.2114374946080765e-05, 'fd_step': 0.001, 'fd_sites_per_condition': 4}. The second Taylor coefficient is checked against the central second difference divided by2, using double arithmetic with explicit native float32 epsilon at four fixed sites for each of39 conditions. This tests implementation, not finite-amplitude accuracy. First-order scores and baseline KL independently replay H391. Coefficients c1/c2, scores, baselines and bootstrap samples are saved. Site-level worsening rates are added by this reporting script; no negative-recovery or worsening cases are excluded.

Uncertainty: 4,000 paired recipient-document/control-draw resamples. Separate simultaneous27 bands at each amplitude cover18 order-specific recovery estimates plus9 gains. These are not one familywise confidence band across all four amplitudes. Same reused panel and fixed donor assignments; no fresh confirmation of second-order claims. Figure shows point estimates.

Only the unit-amplitude99% gate PASSES; the2×95%,4×90%, and4×gain>5pp universal gates FAIL. Second-order recovery ranges99.989–99.99997% at0.5×,99.883–99.9995% at1×,98.653–99.9918% at2×, and53.37–99.863% at4×. The smallest simultaneous lower recovery bounds are99.182% at1×,94.589% at2× and28.490% at4×. The2× point estimates are strong, but the frozen95% criterion still fails and must not be relabeled as passed.

The simultaneous27 band includes the first-order recoveries and the gains as well as the second-order recoveries; its common maximum-deviation radius reflects their variability. At4×, every point gain is positive (10.19–79.88pp), but the smallest simultaneous lower gain is−14.69pp, so universal positive gain above5pp is not established. These conservative bands do not invalidate the point improvements, but they limit the universal claims available from this experiment. No narrower post-hoc band replaces the declared criteria.

At4×, head0 second-order recovery is99.86% across all three families, head1 is95.31–95.54%, and head6 is89.79% actual,53.37% moment,54.57% donor. Compared with H391’s first-order9.92% for actual head6, the consistent second-order term captures much of the coupled error. It is less effective for head6 controls, where H391 identified a strong final-normalization contribution. The intervention family therefore matters even when using a consistent Taylor expansion.

Implementation audits pass: maximum relative error of the second Taylor coefficient versus central second differences is4.21e-5, below the frozen5e-4 audit tolerance. First-order KL and baseline replay H391 within1.8e-15 and8.9e-16 respectively. This separates the predictor’s finite-amplitude limitations from an incorrect derivative implementation. Site-level worsening counts remain in the JSON; positive pooled improvements do not guarantee per-site improvement.

Hypothesis update: coordinated second-order structure explains much more of the finite response than first order, particularly for heads0/1 and at2×. A fixed quadratic approximation is still not uniformly sufficient at4×. This neither identifies semantic units nor compresses the model weights: c1 and c2 are context-and-payload-specific response coefficients calculated from native weights. The remaining head6-control gap is consistent with a final RMS Taylor series being inadequate for large MLP changes, but higher-order input normalization error is an alternative that needs separation.

Next discriminator: retain the second-order MLP prediction a m1+a²m2, but apply final RMS normalization EXACTLY to state+a m1+a²m2 before unembedding. Compare it with the fully polynomial normalized-response predictor tested here, at the same amplitudes. Both agree through second order near zero, while the exact-final-RMS version includes higher-order normalization effects and preserves the normalized state’s geometry. It uses no exact finite target in prediction. If this closes the head6-control gap, the missing final normalization rather than the local MLP response polynomial is implicated; if not, test higher-order input-side response terms. Keep all failures and use fresh documents to confirm any selected extension. Future fresh panels exclude6,336 document hashes. Broad goal remains active.
