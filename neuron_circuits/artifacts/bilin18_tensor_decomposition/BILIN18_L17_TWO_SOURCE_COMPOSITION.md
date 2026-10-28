

## H395 — Do single-source response laws compose?

H394 established a rational identity along one intervention direction. H395 tests simultaneous interventions on actual head pairs (0,1), (0,6), (1,6), at eight frozen signed amplitude combinations on the reused 512-document/2,048-site H390 panel. Only the MLP response changes; the direct residual remains intact. This tests composition within this layer/model and does not establish semantic feature identities.

For directions h and k, let alpha_h=mean(p h)/s0², beta_h=mean(h²)/s0², gamma=mean(h k)/s0², s0²=mean(p²)+epsilon. Define N_h(a)=a*m1_h+a²*(m2_h+2*alpha_h*m1_h), and similarly N_k(b). The joint denominator is q=1+2a*alpha_h+2b*alpha_k+a²*beta_h+b²*beta_k+2ab*gamma. With M0 the intact bias-free bilinear MLP output, C=2B(h,k)/s0²−2gamma*M0, the exact joint change is [N_h(a)+N_k(b)+ab*C]/q. The constant output bias cancels in this response. Apply exact final RMS and native unembedding/cap/softmax for every method.

Five frozen methods: sum_single sums the two separate finite responses; shared_denominator uses (N_h+N_k)/q; norm_cross_only adds only −2ab*gamma*M0 to that numerator; projected_cross adds ab times the Euclidean projection of C onto the four local single-direction vectors; full_joint adds the complete ab*C. The projection is an analytic mixed-vector oracle, not an independently learned coefficient predictor. It tests whether additional output directions matter and does not identify the best subspace under native KL. The omission methods are operational counterfactuals; their differences are not an additive causal decomposition.

Frozen gates: {'exact_kl': True, 'double_identity': True, 'sum_single_99': False, 'shared_denominator_99': False, 'projected_cross_99': False}. Exact formula requires maximum native KL <1e-7 and four-site-per-pair float64 algebra audit error <1e-9. Each approximate-method 99% gate requires all24 simultaneous lower recovery bounds >.99. Each method has its own joint24 band; no familywise claim across all five methods is made. All gates and amplitude pairs were saved before outcomes.

|Pair|Amplitudes|Sum separate|Shared denominator|Norm cross only|Projected cross|Full joint|
|---|---|---:|---:|---:|---:|---:|
|[0, 1]|[1, 1]|99.94619%|99.97013%|99.98191%|99.98773%|100.00000%|
|[0, 1]|[1, -1]|99.91830%|99.96574%|99.97857%|99.98513%|100.00000%|
|[0, 1]|[2, 2]|99.79893%|99.84597%|99.90830%|99.93920%|100.00000%|
|[0, 1]|[2, -2]|99.73124%|99.88254%|99.92567%|99.94825%|100.00000%|
|[0, 1]|[4, 4]|99.23319%|99.00791%|99.41205%|99.63009%|100.00000%|
|[0, 1]|[4, -4]|99.27033%|99.65361%|99.77466%|99.84182%|100.00000%|
|[0, 1]|[1, 4]|99.92003%|99.92777%|99.95698%|99.97219%|100.00000%|
|[0, 1]|[4, 1]|99.78582%|99.81271%|99.88811%|99.92577%|100.00000%|
|[0, 6]|[1, 1]|99.96080%|99.99019%|99.99176%|99.99316%|100.00000%|
|[0, 6]|[1, -1]|99.96442%|99.99350%|99.99417%|99.99540%|100.00000%|
|[0, 6]|[2, 2]|99.84733%|99.94983%|99.95824%|99.96421%|100.00000%|
|[0, 6]|[2, -2]|99.87117%|99.97734%|99.97852%|99.98321%|100.00000%|
|[0, 6]|[4, 4]|99.47793%|99.72849%|99.77400%|99.79150%|100.00000%|
|[0, 6]|[4, -4]|99.56940%|99.92458%|99.92298%|99.94263%|100.00000%|
|[0, 6]|[1, 4]|99.95726%|99.98282%|99.98560%|99.98675%|100.00000%|
|[0, 6]|[4, 1]|99.77110%|99.91998%|99.93326%|99.94451%|100.00000%|
|[1, 6]|[1, 1]|99.72216%|99.80721%|99.72401%|99.89095%|100.00000%|
|[1, 6]|[1, -1]|99.72526%|99.84531%|99.77532%|99.91330%|100.00000%|
|[1, 6]|[2, 2]|98.92947%|99.07013%|98.65911%|99.46331%|100.00000%|
|[1, 6]|[2, -2]|99.02467%|99.39787%|99.07702%|99.64674%|100.00000%|
|[1, 6]|[4, 4]|96.36831%|94.98290%|93.35642%|97.50346%|100.00000%|
|[1, 6]|[4, -4]|96.88199%|97.81524%|96.31453%|98.75890%|100.00000%|
|[1, 6]|[1, 4]|99.66062%|99.58395%|99.43139%|99.81833%|100.00000%|
|[1, 6]|[4, 1]|99.21865%|99.25257%|99.00449%|99.54190%|100.00000%|

Audits: {'max_native_exact_kl': 1.3059097651719107e-11, 'float64_relative_error': 1.1146845094522834e-13, 'float32_relative_error': 0.00041875429451465607, 'min_denominator': 0.4935912489891052, 'cross_projection_relative_residual_quantiles': [0.18051961064338684, 0.5923703908920288, 0.8859410494565964, 0.9944989085197449]}. Independent saved-array audit: {'recovery_max_absolute_error': 0.0, 'projected_cross_max_absolute_replay': [0.0, 0.0, 0.0]}. Signed KL values are retained; assertions permit only tiny numerical negatives. Baseline KL and denominator are positive at all evaluated sites. Raw site scores, mixed vectors, projection coefficients and 4,000 bootstrap recovery arrays are retained.

Uncertainty uses paired document bootstrap resampling, preserving all four sites, source pairs and amplitudes together. Recovery is one minus pooled prediction KL divided by pooled intact-versus-target KL. The panel is reused; these are not fresh confirmation outcomes. No matched random/moment/donor pair controls were included. Model weights and per-context vectors are used directly; there is no model compression or semantic labeling claim.

Outcome: the exact joint identity passes both arithmetic gates (maximum native KL 1.30591e-11; double-precision relative response error 1.11468e-13). All three frozen approximate-method 99% gates fail across the full amplitude grid. These failures are retained rather than replacing the threshold with one suggested by the results.

At the ordinary signed unit combinations (1,1) and (1,−1), summing the separate laws recovers 99.722–99.964% of the native KL effect. Its minimum simultaneous lower bound at those combinations is 99.133%, using the already-defined band over all24 cases. Thus substantial within-source nonlinearity can coexist with weak cross-source interaction at ordinary removal amplitudes. This is a secondary reading of the frozen grid, not a new gate or independent confirmation.

The difficult pair is heads1/6 at (4,4): recovery is 96.368% for separate laws, 94.983% for the shared denominator, 93.356% for normalization-cross-only, and 97.503% for the projected mixed term. Exact mixed response restores numerical agreement. Correcting only one coupling can therefore worsen the approximation; normalization and bilinear contributions can compensate. Adding the complete mixed response matters here, but the deficit is much smaller than the previously observed failure of first-order prediction along one source at large amplitude.

The mixed vector's Euclidean relative residual after projection onto the four single-direction vectors has median0.59237, 95th percentile0.88594 and maximum0.99450. These are norm fractions, not squared energy fractions. Mixed responses often introduce substantial new directions, while their effect on native output distributions can remain small. Projected-cross recovery spans97.503–99.995%; its universal99% gate fails. This does not prove the four-vector span cannot achieve99% under a native-KL-optimized fit: the implemented projection minimizes Euclidean mixed-vector error. Nor does it prove any new direction is a semantic feature. Local independence, output-metric redundancy and limited intervention magnitude remain competing explanations.

The maximum single-document share of baseline KL over the24 cases is4.704%. This supplementary diagnostic does not exclude heavy error concentration or masking within documents. Pair controls, broader head coverage, fresh documents and earlier layers remain needed for broader claims.

### Twenty hypotheses after the composition test

These are competing, sometimes compatible explanations; priorities are qualitative, not posterior probabilities. The present experiment is restricted to three actual L17 head pairs. Proposed follow-ups below are not preregistered results.

|#|Hypothesis and reason|Discriminating experiment|Priority/evidence|
|---|---|---|---|
|1|Single-source rational laws are useful local computational units because cross-source interaction is weak.|Confirm sum-of-laws prediction on fresh documents and more head pairs.|Leading operational hypothesis; not semantic identity.|
|2|Weak interaction is generic at these perturbation norms, rather than learned source separation.|Joint moment/donor controls matched to actual norms and angles.|Leading alternative; current actual-only experiment cannot distinguish.|
|3|Different head output subspaces have unusually weak bilinear coupling.|Compare cross-tensor energy between source subspaces against matched random subspaces.|Plausible; contextual results alone are insufficient.|
|4|Normalization coupling explains most composition error.|Compare shared-denominator and separate-denominator models over larger panels and amplitude scales.|Directly probed, but improvements need not be universal.|
|5|The mixed numerator adds directions absent from the single laws.|Measure mixed-vector projection residual and behavioral error after projection.|Directly probed; vector residual and KL importance must be distinguished.|
|6|Novel mixed directions are mostly invisible to the current output distribution.|Compare Euclidean residual energy with probability-weighted output energy.|High-information next diagnostic.|
|7|Mixed response is redundant specifically under the native output metric.|Fit mixed vector in the four-vector span using recipient probability geometry, then score finite outputs.|Plausible; Euclidean projection does not settle this.|
|8|Apparent additivity comes from domination by one source's effect.|Stratify by single-source effect balance and inspect symmetric pairs.|Important alternative.|
|9|Cancellation makes opposite-sign combinations unusually fragile.|Compare matched effect sizes and denominator behavior across signs.|Signed grid probes this; rare cancellation remains possible.|
|10|Aggregate recovery hides a small set of strongly interacting documents.|Inspect per-document excess error and effect-weighted tails; confirm selected patterns on fresh data.|Important robustness check.|
|11|Coupling emerges only when many sources change together.|Extend exact quadratic numerator to 3–9 heads and compare pairwise versus independent approximations.|Plausible; pairwise success is not many-source success.|
|12|Three selected heads are atypical.|Test all36 head pairs with frozen metrics and joint uncertainty.|Strong scope limitation.|
|13|L17 is more separable than earlier layers.|Repeat matched actual/control pair tests at earlier depths.|High priority for model-wide claims.|
|14|Exact final RMS hides substantial interaction in raw MLP space.|Compare raw response norms, final normalized response and native KL.|Plausible metric dependence.|
|15|Logit capping suppresses mixed-response consequences.|Evaluate a clearly labeled uncapped readout diagnostic alongside native scoring.|Mechanistic diagnostic; altered metric is not native behavior.|
|16|Probability concentration makes many response directions interchangeable.|Stratify joint errors by intact entropy and compare token-resolved effects.|Supported motivation from earlier metric experiments; not tested here.|
|17|Local laws transfer between similar contexts.|Predict coefficients from independent context clusters and evaluate new documents.|Unresolved; current per-context calculations do not establish transfer.|
|18|A small number of native bilinear factors generate each trajectory and its mixed terms.|Measure factor contributions with cancellation-aware interventions and held-out reconstruction.|Unresolved route toward mechanistic units.|
|19|Source-associated laws align with semantic functions.|Test controlled linguistic contrasts and matched lexical/context controls after establishing transfer.|Unresolved; head identity alone is not a semantic label.|
|20|The useful unit is a context-conditioned response operator, not a fixed residual vector.|Compare prediction and description cost of operators, fixed bases and native factors on unseen contexts and simultaneous edits.|Leading broad framing, still requires predictive compression.|

Narrowing: prioritize hypotheses 1/2 (learned versus generic additivity), 5/6/7 (new directions versus behaviorally relevant directions), and 8/10 (aggregate masking). The next experiment should pair actual interventions with norm/angle-matched source controls, and include effect-balanced and document-tail summaries. That can discriminate several explanations at once without claiming semantic features from a successful algebraic identity. Fresh confirmation should follow any selected contrast; reusing this panel is suitable for diagnosis, not independent confirmation.
