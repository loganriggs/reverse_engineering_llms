

## H393 — Second-order MLP prediction with exact final RMS

Keep H392’s local MLP coefficients m1,m2, but predict RMS(state+a m1+a²m2)−RMS(state) instead of the normalized-response Taylor polynomial a c1+a²c2. Both agree through second order mathematically; the new version retains higher-order final-normalization effects while leaving the input-side MLP approximation fixed. It has no access to the actual finite response during prediction and uses no fitted coefficients or step selection. Full native weights, intact context and source payload are still required.

Same H390512-document panel, amplitudes0.5/1/2/4, heads0/1/6 and39 response conditions. Recompute first-order and full polynomial second-order controls. All outputs receive native unembedding, cap and softmax; no extra normalization follows the hybrid’s normalized output. Finite native response appears only in evaluation.

|Head|Family|Polynomial1|Hybrid1|Polynomial2|Hybrid2|Polynomial4|Hybrid4|
|---|---|---:|---:|---:|---:|---:|---:|
|0|actual|1.0000|1.0000|0.9999|0.9999|0.9986|0.9975|
|0|moment|1.0000|1.0000|0.9999|0.9999|0.9986|0.9983|
|0|donor|1.0000|1.0000|0.9999|0.9998|0.9986|0.9982|
|1|actual|0.9999|0.9998|0.9976|0.9958|0.9531|0.9203|
|1|moment|0.9998|0.9998|0.9973|0.9958|0.9554|0.9317|
|1|donor|0.9999|0.9998|0.9974|0.9958|0.9545|0.9284|
|6|actual|0.9995|0.9987|0.9919|0.9768|0.8979|0.6668|
|6|moment|0.9989|0.9998|0.9877|0.9955|0.5337|0.8784|
|6|donor|0.9988|0.9998|0.9865|0.9954|0.5457|0.8558|

Frozen gates: {'hybrid_unit99': True, 'hybrid_two95': True, 'hybrid_four90': False, 'head6_controls_gain10': False}. Require all9 hybrid lower recoveries >99% at1×, >95% at2×, >90% at4×. The targeted gain criterion requires both head6 control-family lower gains over polynomial second order >10pp at4×, using the simultaneous9-family gain band. The head6 focus was selected from H391–392 before this experiment.

Audits: {'h392_both_orders_replay_max_absolute': 0.0, 'baseline_h391_replay_max_absolute': 8.881784197001252e-16, 'first_order_h391_replay_max_absolute': 1.7763568394002505e-15, 'second_taylor_coefficient_fd_max_relative': 4.2114374946080765e-05, 'fd_step': 0.001, 'fd_sites_per_condition': 4}. First and second polynomial score arrays and baseline KL independently replay H392. The coefficient finite-difference audit checks the unchanged local derivatives, not the accuracy of the finite hybrid. Individual worsening rates are retained in JSON; no failures are excluded. Saved arrays include all three prediction scores and baselines. Independent site-audit fields are added by this reporting script.

Uncertainty: 4,000 paired recipient-document/control-draw resamples. At each amplitude, one simultaneous9 band covers hybrid recovery across all groups, and a separate simultaneous9 band covers hybrid-minus-polynomial gains. These bands were specified before outcomes and are not a single familywise band across amplitudes/statistic families. Unlike H392, they do not include first-order recovery variability in the hybrid confidence radius. H392’s failed gates remain unchanged; differences in gate outcomes must not be attributed solely to changed predictions. This reused panel is not fresh confirmation. Figures show point estimates.

The hybrid unit99 and2×95 gates PASS; the4×90 and targeted head6-control gain10 gates FAIL. Hybrid recovery ranges99.866–99.9991% at1×,97.676–99.9852% at2×, and66.68–99.827% at4×. Minimum simultaneous lower hybrid recoveries are99.821%,96.882%, and56.395%, respectively. The changed confidence-band family is explicit above; H392's failed2× gate remains a failure, and passing here does not mean the hybrid is pointwise better than the polynomial predictor.

The mechanism is mixed. At4×, head6 moment/donor controls improve from53.37/54.57% polynomial recovery to87.84/85.58% hybrid recovery, gains34.47/31.00pp. Their simultaneous gain intervals are[12.44,56.49]pp and[8.98,53.03]pp. Both improvements are positive under the joint9 band, but the donor lower bound misses the10pp threshold, so the targeted gate fails. Exact final normalization explains a substantial portion, not all, of these control errors.

Actual head6 moves in the opposite direction: recovery drops from89.79% to66.68%, a23.11pp loss, with hybrid-minus-polynomial interval[−45.14,−1.08]pp. Head1 loses2.37–3.28pp at4×, and head0 loses0.03–0.12pp. Thus keeping one stage exact can remove compensating errors in the approximate composition. The result does not imply exact RMS is harmful to the true model; it says exact RMS applied to an imperfect second-order MLP response need not outperform a jointly truncated normalized-response polynomial.

All replay and derivative audits pass. The two previous prediction grids replay H392 exactly. The only altered predictive stage is final RMS; the finite target is still used solely for scoring. Source-family-dependent improvements and losses are therefore informative, not a numerical replay difference. No per-site worsening cases are excluded.

Hypothesis update: a uniformly sufficient second-order MLP plus exact-final-normalization description is disfavored at4×. Input-side higher-order effects remain relevant, and compensation between input-side and output-side errors matters. A third-order fit is possible, but the known bilinear/RMS architecture suggests a stronger algebraic test with no learned higher-order coefficients.

Derived next hypothesis (not yet tested): along a fixed source-removal direction h=−payload, let s0²=mean(pre²)+epsilon, alpha=mean(pre*h)/s0² and beta=mean(h²)/s0². The input-normalized bilinear MLP has a quadratic numerator divided by q(a)=1+2 alpha a+beta a². If m1,m2 are its first two local Taylor response coefficients, algebra gives the exact real-arithmetic MLP response along that path:

Delta_MLP(a) = [a m1 + a²(m2+2 alpha m1)] / [1+2 alpha a+beta a²].

Apply final RMS exactly after adding this response to the intact state. This expression predicts all finite amplitudes using two context/payload-specific vectors and two input-geometry scalars; it does not require evaluating the finite input-normalized MLP at prediction time. It follows from the homogeneous bilinear architecture, not from newly learned semantic features, and does not compress the global weight tensor. In particular, source angle/norm-matched controls share these input-geometry scalars at a site while their response vectors differ.

Next experiment: implement and numerically validate this rational response formula against native interventions, including amplitudes beyond the tested Taylor range. If it succeeds, higher Taylor orders along a single source direction are determined by the first two coefficients and normalization geometry rather than independent response components. Check finite-precision stability and the scalar denominator; keep the direct residual fixed as in all MLP-only experiments. The rational prediction is an unverified consequence here, not a completed result. Fresh-document exclusions remain6,336 hashes; broad goal remains active.
