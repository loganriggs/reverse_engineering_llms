

## H408 — Rank native factors by their contribution to output sensitivity

Discovery uses H400512 documents; evaluation uses H406512 documents. Panels are disjoint from factor selection but have prior research use, so this is a screening experiment, not fresh confirmation. Actual head6 MLP-path edits only, amplitudes1 and4, ranks16/64/256. All H405 contrast, individual-energy and six random orders are preserved. Native Down vectors are never refitted.

The new signed attribution for factor j is phi_j D_jᵀ F g, where g is the complete head6 MLP derivative and F is the native output Fisher metric pulled through final RMS normalization, unembedding, capped logits and softmax. Its sum is gᵀFg. Scores are averaged over discovery sites, sorted descending and saved before evaluation. This is first-order signed attribution, not an optimization of subset reconstruction; cancellation can make some scores negative. Float64 pullbacks preserve the native float32 RMS epsilon.

For native finite change r and retained-factor change rS, score KL(native edited || RMS(s+rS)) and KL(RMS(s+r−rS) || intact). The former measures retained-response prediction; the latter measures effect remaining under a baseline-preserving activation clamp. Normalize pooled KLs by the pooled original effect, not by individual-site ratios. The direct residual path stays fixed. Clamp fractions are not additive mediation fractions.

Frozen numerical gates: Fisher closure<1e-8 and native effect replay<1e-10. Frozen primary gates at rank64/unit amplitude: recovery joint lower>.9; clamp remaining joint upper<.2; recovery improvement over contrast64 joint lower>.05. Four thousand paired document bootstrap samples form one joint3 maximum-absolute-deviation band. No gate was changed after outcomes.

|Primary statistic|Estimate|Joint95% interval|
|---|---:|---|
|fisher64_unit_recovery|0.655703|[0.611321484670886, 0.7000845320193098]|
|fisher64_unit_clamp_remaining|0.337099|[0.2927170362582783, 0.38148008360670194]|
|fisher64_minus_contrast64_unit_recovery|-0.055470|[-0.09985199300437513, -0.011088945655951407]|

Gates: {'fisher_closure': True, 'native_replay': True, 'native_recovery90': False, 'clamp_remaining20': False, 'improvement05': False}.

|Ranking|Factors|Amplitude|Native KL recovery|Clamp remaining fraction|Raw vector recovery|
|---|---:|---:|---:|---:|---:|
|fisher|16|1|0.484263|0.504855|0.746816|
|fisher|16|4|0.509832|0.451148|0.765128|
|fisher|64|1|0.655703|0.337099|0.892177|
|fisher|64|4|0.611807|0.350303|0.914314|
|fisher|256|1|0.757383|0.238000|0.920928|
|fisher|256|4|0.742423|0.236992|0.943264|
|contrast|16|1|0.296272|0.695029|0.813486|
|contrast|16|4|0.351042|0.589525|0.799280|
|contrast|64|1|0.711173|0.290479|0.967059|
|contrast|64|4|0.722986|0.270579|0.974964|
|contrast|256|1|0.803387|0.196747|0.978187|
|contrast|256|4|0.798229|0.192493|0.983791|
|individual_energy|16|1|0.316423|0.672953|0.827207|
|individual_energy|16|4|0.359200|0.584595|0.812108|
|individual_energy|64|1|0.712161|0.288965|0.969498|
|individual_energy|64|4|0.720207|0.270978|0.976033|
|individual_energy|256|1|0.845411|0.153256|0.984799|
|individual_energy|256|4|0.830131|0.157784|0.988000|

Audits: {'fisher_attribution_relative_closure': 3.110964647102098e-10, 'direct_clamp_state_absolute_error': 6.67572021484375e-06, 'native_effect_replay_max_absolute': 0.0, 'contrast_energy_random64_prediction_replay': 9.367506770274758e-16, 'contrast_energy_random64_clamp_replay': 8.674701582056521e-16}. Independent saved-array checks: {'mean_metrics_replay': True, 'control_orders_unchanged': True, 'ranking_matches_saved_scores': True, 'pooled_Fisher_score_closure': 4.155667238637022e-16, 'top64_overlap_contrast': 47, 'top64_overlap_energy': 44}. Nonprimary rank/amplitude comparisons above are descriptive. No rotated-source specificity, permanent compression, global feature ontology or semantic labeling is established.

Outcome: both numerical gates pass, all three scientific gates fail. Fisher64 unit recovery is0.655703 with joint interval[0.611321,0.700085]; baseline-preserving clamp remaining fraction is0.337099,[0.292717,0.381480]. Its recovery difference from contrast64 is−0.055470,[−0.099852,−0.011089]. Thus this particular signed native-output ranking is worse at64 factors on the held-out panel; merely changing the importance metric does not resolve the unexplained output effect. The comparison is paired, but conditional on these fixed discovery rankings and this previously studied evaluation panel.

The rank dependence is informative: Fisher16 has descriptive unit recovery0.4843 versus contrast16 0.2963 and individual-energy16 0.3164. By256 factors the order reverses: Fisher0.7574, contrast0.8034, individual-energy0.8454. At4× amplitude Fisher64 falls to0.6118 versus contrast0.7230. These nonprimary comparisons are descriptive, not new confirmatory tests. Fisher64 overlaps47 of contrast64 and44 of individual-energy64 factors. Of4608 discovery signed scores,1656 are negative; the top64 scores sum to77.08% of total discovery Fisher energy. That attribution fraction is not subset reconstruction fidelity, and it must not be interpreted as recovered output effect.

Update to the twenty-hypothesis tournament: #3 remains supported for raw head6 response sparsity, while #4 remains viable for native-output effect support. #16 (output metric matters) remains well supported, but the specific remedy of signed Fisher ranking is weakened. #5 (factor cancellation/cooperation) rises in priority because signed attribution measures correlation with the whole response, not the error left by removing a collection. #12 (output alignment) and #13 (context-dependent support) remain alternatives; no semantic evidence for#19 was added.

Next highest-information experiment: evaluate the SAME frozen subsets in the local Fisher tangent metric on both discovery and evaluation panels. Save retained energy, omitted energy, cross term and signed attribution with their exact sum identities. This separates three explanations without selecting another ranking: (a) poor local subset reconstruction already on discovery implicates attribution-versus-subset cancellation; (b) good discovery but poor held-out tangent reconstruction implicates context transfer; (c) good held-out tangent but poor finite KL reconstruction implicates finite-edit nonlinearity. Compare ranks16/64/256 and preserve both raw-energy baselines and six random orders. Freeze thresholds before scoring. Only after this diagnostic should we choose between residual-aware subset selection, context-dependent selection, or finite-response ranking. Do not infer that Fisher geometry itself is ineffective from the failure of this single ranking rule.

The broad feature/computation-unit goal remains open. No fresh documents were consumed; the next fresh panel must still exclude7872 complete-document hashes. No background jobs remain.
