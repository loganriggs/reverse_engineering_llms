

## H391 — Intervention amplitude and three sources of curvature

Prespecified amplitudes0.5,1,2,4 multiply each source payload on the H390512-document panel. Heads0/1/6, actual plus six moment and six donor controls, four positions. For each amplitude evaluate all eight combinations of: exact versus first-order input RMS (bit4); retain versus omit the bilinear quadratic response B(u,u) (bit2); exact versus first-order final RMS (bit1). Here u is the selected normalized-input change, and MLP response is2B(x,u)+B(u,u) when quadratic is included. Direct residual pre stays intact, matching the MLP-only intervention target. Full native capped-vocabulary KL evaluates unrestricted predictions; no low-rank projection or step safeguard.

Mask0 is the previously confirmed first-order predictor; mask7 must reproduce the native target. At amplitude1 native responses and linear predictions replay H390. Amplitudes above1 overshoot removal of the original source contribution and are controlled extrapolations, not ordinary source ablations. The population of contexts is unchanged, but baseline intervention KL changes with amplitude.

|Head|Family|Linear0.5|Linear1|Linear2|Linear4|
|---|---|---:|---:|---:|---:|
|0|actual|0.9983|0.9925|0.9641|0.7991|
|0|moment|0.9961|0.9838|0.9320|0.7338|
|0|donor|0.9961|0.9839|0.9322|0.7318|
|1|actual|0.9975|0.9887|0.9448|0.6811|
|1|moment|0.9967|0.9856|0.9327|0.6641|
|1|donor|0.9969|0.9865|0.9360|0.6685|
|6|actual|0.9932|0.9672|0.8228|0.0992|
|6|moment|0.9900|0.9678|0.9058|0.4108|
|6|donor|0.9898|0.9676|0.9056|0.4439|

|Head|Family|Linear|Output only|Quadratic only|Quadratic+output|Input only|Input+output|Input+quadratic|All exact|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|actual|0.7991|0.8036|0.9030|0.9038|0.9555|0.9549|0.9997|1.0000|
|0|moment|0.7338|0.7210|0.8705|0.8608|0.9486|0.9455|0.9993|1.0000|
|0|donor|0.7318|0.7202|0.8700|0.8609|0.9479|0.9450|0.9994|1.0000|
|1|actual|0.6811|0.6186|0.8880|0.8362|0.8667|0.8112|0.9698|1.0000|
|1|moment|0.6641|0.6040|0.8664|0.8186|0.8770|0.8354|0.9763|1.0000|
|1|donor|0.6685|0.6131|0.8700|0.8239|0.8743|0.8311|0.9756|1.0000|
|6|actual|0.0992|-0.0124|0.6973|0.5499|0.7974|0.3457|0.8290|1.0000|
|6|moment|0.4108|0.7231|0.4904|0.8898|0.5034|0.8242|0.5466|1.0000|
|6|donor|0.4439|0.7289|0.5022|0.8947|0.5234|0.8152|0.5540|1.0000|

The second table is amplitude4. Shapley contributions average each correction over all six orders of adding the three corrections; they sum to R(all exact)−R(linear). Shares divide by that positive total. Negative contributions and shares above1 are retained: correction terms can compensate. These are attributions of prediction error under native KL, not a unique causal ontology of network components.

Frozen gates: {'linear_unit90': True, 'linear_four90': False, 'input_majority_four': False, 'quadratic_majority_four': False, 'output_majority_four': False}. Linear90 requires all9 lower recovery bounds >90% at the specified amplitude. The three competing majority gates require every amplitude4 share lower bound >50% for that correction.

Audits: {'all_exact_corner_max_kl': 2.3391370878287927e-11, 'unit_native_response_max_absolute': 0.0, 'unit_linear_prediction_max_absolute': 0.0, 'payload_over_pre_norm_quantiles': [0.038613537326455116, 0.09767808020114899, 0.14958487451076508, 0.23285765945911407]}. Independent saved-array audit: {'h390_unit_baseline_replay_max_absolute': 2.5326962749261384e-16, 'h390_unit_linear_kl_replay_max_absolute': 2.5847379792054426e-16, 'shapley_sum_identity_max_absolute': 1.1102230246251565e-16}. Input artifact hashes verified; all finite scores checked and no failing or negative-recovery cases excluded. Native-target exactness is an algebraic control, not evidence of approximation quality. Full per-site eight-corner scores, baselines, source/residual norm ratios, bootstrap samples and all confidence bands are retained. Independent audit fields are added by this reporting script.

Uncertainty: 4,000 paired recipient-document/control-draw bootstrap resamples. At EACH amplitude separately, simultaneous72 bands cover all head/family/corner recoveries, and simultaneous27 bands cover all head/family/factor shares. These are not one familywise band over all four amplitudes or over both statistic families. Fixed donor assignments, reused panel, no fresh-document confirmation of the amplitude findings. Figures show point estimates.

The unit-amplitude linear90 gate PASSES; the amplitude4 linear90 gate and all three universal majority gates FAIL. Unrestricted first-order recovery spans98.98–99.83% at0.5×,96.72–99.25% at1×,82.28–96.41% at2×, and9.92–79.91% at4×. The strong unit-scale result has a clear finite-amplitude boundary. Confidence bands are wider at4×; point ranges should not be mistaken for simultaneous bounds on every group.

At4× the correction pattern is heterogeneous. Head0 has input-RMS Shapley shares62.49–66.44%, bilinear-quadratic35.87–36.70%, output-RMS−2.31 to+0.81%. Head1 mixes input50.86–56.06% and quadratic51.99–58.12%, with negative output shares−8.98 to−7.61%. Head6 ACTUAL has input51.56%, quadratic57.33%, output−8.89%, whereas head6 moment/donor controls have output shares63.69/64.32%. Their output-share joint intervals are[50.34%,77.04%] and[50.97%,77.67%], respectively. Head0 moment/donor input-share intervals also exceed50% ([53.09%,79.79%],[52.79%,79.49%]). These are subgroup observations within the prespecified simultaneous27 intervals, not replacements for the failed universal gates.

The eight corners reveal why a single attribution is inadequate. For head6 actual at4×, linear recovery is9.92%; exact input RMS alone raises it to79.74%, quadratic alone to69.73%, but exact output RMS alone lowers it to−1.24%. Exact input+quadratic with linear output reaches82.90%; adding exact output to that correct MLP response then closes the remaining17.10 points. For head6 controls, exact input+quadratic with linear output recovers only54.66/55.40%, and final RMS supplies the remaining45.34/44.60 points. Thus a negative averaged output share does not mean final normalization is intrinsically harmful: its effect depends strongly on whether the incoming MLP response is correct.

For head0, exact input+quadratic already recovers99.93–99.97% at4× even with linear output RMS; head1 reaches96.98–97.63%. The control/source distinction and head dependence are therefore substantial. An account in terms of only bilinear self-interaction, only input normalization, or only output normalization misses the observed dependence on the other terms.

Scale context: unit payload/pre-residual norm ratios have median0.0386,95th percentile0.0977,99th percentile0.1496 and maximum0.2329. At4× these ratios multiply by four, reaching0.9314 at the largest site. These are genuine finite extrapolations. All-exact maximum KL is2.34e-11, and unit native responses and linear predictions replay exactly. Unit score arrays independently reproduce H390. No negative-recovery cases were dropped.

Supplementary H390 audit, performed while this factorial ran: deleting any single document changes the practical-path loss versus the oracle by at most0.178 percentage points; the largest document share of baseline intervention KL is4.33%. Unrestricted-linear individual failures occur at0–0.293% of sites by group and account for at most0.00154% of baseline effect mass. These are exploratory influence diagnostics with fixed donor assignments, not new acceptance criteria. Full details are in bilin18_l17_linear_response_confirmation_influence_audit.json.

Hypothesis update: local first-order computation predicts the original intervention family, but its extension to larger interventions requires coupled curvature terms. Which term matters depends on source family and head. Shapley shares describe this prediction-error experiment, not unique semantic units. Exact arithmetic decompositions reproduce the native model by construction and should not be counted as learned compression.

Next discriminator: a CONSISTENT second-order Taylor predictor including input RMS second-order change, the bilinear quadratic response, and final RMS second-order change together. If u1,u2 are the first two input-normalization Taylor coefficients, m1=2B(x,u1), m2=2B(x,u2)+B(u1,u1); the normalized response predictor is a Drms(s,m1)+a²[Drms(s,m2)+0.5 D²rms(s)[m1,m1]]. Compare this input-derived predictor with first order across the same frozen amplitude ladder. It uses local derivatives and native weights, without consulting the exact finite response for prediction. This tests whether coordinated local curvature explains the failure, rather than adding isolated exact corrections that can destroy compensating errors. Follow any selected amplitude claim with fresh confirmation. Fresh panels must exclude6,336 prior complete-document hashes; broad goal remains active.
