

## H396 — Is near-additivity specific to actual source pairs?

Compare the sum of two context-specific single-source rational response laws against the native simultaneous MLP-only intervention. Three L17 head pairs (0,1),(0,6),(1,6), four signed amplitude pairs (1,1),(1,−1),(4,4),(4,−4), actual payloads and six moment/six donor control pairs. Reuse all512 H390 documents and2,048 sites. The same control-draw index is used for both sources. Moment noise is independently drawn per head using the established full-nine-head generator. Donor controls use the same donor document for both sources before separate angle matching. The direct residual is intact, and final RMS/unembedding/cap/softmax are exact. No finite target is used by the sum predictor.

Controls preserve each source norm and dot product with the pre-MLP residual within its original head span. They do not preserve mutual source angle, full joint geometry, raw response size or output-effect size. Thus this tests actual-versus-control pairing under those constraints, not a pure semantic intervention or a complete separation of geometry and learned coupling. Frozen local coefficients from H394 are reused; the regenerated payload geometry is checked against their saved scalars.

Frozen gates: {'joint_formula_audit': True, 'actual_replay': True, 'control_unit99': True, 'actual_advantage': False, 'one_point_equivalence': False}. Full-panel relative float32 error of the exact joint formula must be<.01, and actual-case baseline/prediction KL replay must agree with H395 within1e-10. Control unit recovery gate requires all12 joint lower bounds>.99. Actual advantage requires all24 actual-minus-control joint lower bounds>0. One-percentage-point equivalence requires all24 intervals to lie within[−.01,.01]. These are universal criteria; their failure does not imply the opposite universal claim.

|Pair|Amplitudes|Actual recovery|Moment recovery|Donor recovery|Actual−moment (joint95% CI), pp|Actual−donor (joint95% CI), pp|
|---|---|---:|---:|---:|---:|---:|
|[0, 1]|[1, 1]|99.9462%|99.9478%|99.9475%|-0.0016 [-0.6199,0.6167]|-0.0013 [-0.6196,0.6170]|
|[0, 1]|[1, -1]|99.9183%|99.9498%|99.9499%|-0.0314 [-0.6498,0.5869]|-0.0316 [-0.6499,0.5867]|
|[0, 1]|[4, 4]|99.2332%|98.8976%|98.6900%|0.3356 [-0.2827,0.9539]|0.5432 [-0.0751,1.1615]|
|[0, 1]|[4, -4]|99.2703%|99.3939%|99.3519%|-0.1236 [-0.7419,0.4947]|-0.0816 [-0.6999,0.5367]|
|[0, 6]|[1, 1]|99.9608%|99.9883%|99.9898%|-0.0275 [-0.6458,0.5908]|-0.0290 [-0.6473,0.5893]|
|[0, 6]|[1, -1]|99.9644%|99.9906%|99.9920%|-0.0261 [-0.6444,0.5922]|-0.0276 [-0.6459,0.5907]|
|[0, 6]|[4, 4]|99.4779%|99.7874%|99.8060%|-0.3095 [-0.9278,0.3088]|-0.3280 [-0.9464,0.2903]|
|[0, 6]|[4, -4]|99.5694%|99.8644%|99.8858%|-0.2950 [-0.9133,0.3233]|-0.3164 [-0.9347,0.3019]|
|[1, 6]|[1, 1]|99.7222%|99.8497%|99.8544%|-0.1275 [-0.7458,0.4908]|-0.1322 [-0.7505,0.4861]|
|[1, 6]|[1, -1]|99.7253%|99.8826%|99.8860%|-0.1573 [-0.7756,0.4610]|-0.1608 [-0.7791,0.4575]|
|[1, 6]|[4, 4]|96.3683%|97.5736%|97.5331%|-1.2053 [-1.8236,-0.5870]|-1.1648 [-1.7831,-0.5465]|
|[1, 6]|[4, -4]|96.8820%|98.5437%|98.6132%|-1.6617 [-2.2800,-1.0434]|-1.7312 [-2.3495,-1.1129]|

Audits: {'actual_replay_max_absolute': 9.150661041821737e-16, 'full_joint_max_float32_relative_error': 0.0007051649736240506, 'saved_single_geometry_max_absolute_error': 0.0}. Recovery independently recomputed from saved site arrays to tolerance1e-12. All native baseline effects and exact joint denominators are positive; numerical KL negatives are bounded below by−1e-10.

Uncertainty:4,000 paired recipient-document/control-draw bootstrap samples, preserving the joint source pairing within each draw. Separate simultaneous bands cover all36 recoveries and all24 actual-minus-control differences. Fixed donor assignments and reused documents limit inference. The report retains descriptive raw-response-balanced strata, document influence, worse-than-intact sites and source-pair-angle quantiles in the machine-readable artifact. Balance strata use2min(norm(r_h),norm(r_k))/(norm(r_h)+norm(r_k))>=.8; they are not matched KL strata and do not constitute new gates.

Outcome: both arithmetic checks and the control unit-amplitude99% gate pass. Actual replay differs from H395 by at most9.15e-16; saved single-source geometry replays exactly; the full joint formula's maximum float32 relative response error is0.000705. Actual-advantage and universal one-percentage-point-equivalence gates fail.

Control unit-amplitude recovery ranges99.850–99.992%, with minimum simultaneous lower bound99.238%. Near-additivity therefore also occurs for the matched control sources. This weakens the interpretation that high ordinary-amplitude additivity identifies special natural source units. It does not prove that actual and control source laws are equivalent at every amplitude.

There are no positive actual-minus-control contrasts whose joint interval excludes zero. Four contrasts significantly favor controls: heads1/6 at both4× signed combinations against both control families. At(4,4), actual recovery is96.368%, moment97.574%, donor97.533%; actual-minus-control joint intervals are[−1.824,−0.587] and[−1.783,−0.547] percentage points. At(4,−4), actual96.882%, moment98.544%, donor98.613%; the differences are−1.662 and−1.731 percentage points, and both joint intervals remain below−1 percentage point. Other contrasts remain uncertain under the joint24 band.

Crucial scale qualification: for heads1/6 at(4,4), mean prediction KL is0.001653 for actual sources versus0.002662/0.002707 for moment/donor controls. The controls have larger absolute approximation error. Their baseline joint effects are also much larger: mean KL0.10970/0.10974 versus0.04553 for actual sources. Thus controls' better percentage recovery is not evidence of smaller absolute interactions. Source norm and residual-angle matching do not match behavioral effect size. Mutual source-angle distributions also differ; those quantiles are saved, not assumed matched.

The raw-response-balanced subset further qualifies aggregate conclusions. For heads1/6 at(4,4), actual recovery is96.161% versus controls95.796/96.000%, reversing the point ordering seen in the full aggregate. At(4,−4), actual balanced recovery falls to93.424%, versus controls95.931/95.755%. These subsets retain different baseline-effect fractions and have no confirmatory intervals; they motivate matching or stratification, rather than establishing a corrected causal comparison. The largest leave-one-document change anywhere is0.174 percentage points. At worst,0.635% of sites are worse than leaving the model intact; in the actual heads1/6 opposite-sign4× case, these sites carry only0.019% of baseline KL mass. The result is not uniformly safe at every site, and the worst-site outcomes are retained.

Hypothesis update: the generic-small-intervention explanation remains strong; special actual-source additivity is weakened; behavior-metric/effect-size explanations gain priority. Evidence for mixed directions outside the single-source span remains intact from H395, but the importance of those directions depends on the readout and normalization. Raw-response balance can change comparative conclusions, so universal claims of learned source independence are unwarranted.

Next high-information experiment: separate interaction size at three stages—raw MLP output, final-normalized residual, and native output distribution—using explicit common scales as well as intervention-relative recovery. Compare actual and matched controls with source-balance and mutual-angle summaries. This can distinguish intrinsically weak bilinear interactions from normalization/readout suppression and denominator-driven recovery changes. Use the same fixed head pairs initially for diagnosis, then fresh documents and broader pairs for any selected claim. A recipient-metric fit of the mixed response in the single-source span remains a separate test of whether novel Euclidean directions are behaviorally necessary; it must be labeled an oracle, not feature prediction.
