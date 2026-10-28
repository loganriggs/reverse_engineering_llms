

## H406 — Finite head6 responses and intact-output factor ablations

Fresh512 documents exclude7,360 complete-document hashes from all previous completed panels; independent document/window reconstruction verifies zero overlap and7,872 total unique completed-panel documents. Freeze the H405 contrast64, energy64 and all six random64 sets without reselection or fitting. Use actual head6 and six new full-Gram-preserving rotation controls at amplitudes1 and4. Edit the MLP input path only: the direct residual remains fixed, as in prior source-response experiments.

Let intact raw final state be s, native finite MLP change r, intact contribution of the selected factors f0, and their finite change rS. The native target is finalRMS(s+r). Retained-response prediction is finalRMS(s+rS): it keeps the intact baseline and tests how much of the intervention response the factors explain. The genuinely factor-deleted baseline is finalRMS(s−f0), and its edited state is finalRMS(s−f0+r−rS). These are different counterfactuals; retained prediction is not the same as keeping only those factors in an intact model.

All states pass through native unembedding, logit cap and softmax. Measure KL(native edited||retained prediction), native source effect KL(native edited||intact), and remaining source sensitivity KL(deleted edited||deleted baseline). Also retain KL(native edited||deleted edited), which combines baseline and response errors. Separately score KL(intact||factor-deleted intact) and actual next-token NLL change using the next corpus token at each sampled position. A positive NLL change denotes worse predictive performance on this panel. KL output displacement alone is not a task-performance metric.

Frozen gates: {'raw_unit90': True, 'native_unit90': False, 'remaining_unit20': False, 'intact_loss02': False}. For contrast64 actual unit removal, require raw response fidelity>.9, native-KL recovery>.9, remaining-effect fraction<.2 and intact NLL increase<.02 nats at simultaneous bounds. Four thousand paired recipient-document bootstrap samples produce one max-absolute-deviation joint4 band over the four preselected statistics (recovery/fraction units and NLL in nats). The primary cases are actual sources only; rotated and random-set curves are descriptive here. No across-family or amplitude-wide gate is implied.

|Primary statistic|Estimate|Joint95% interval|
|---|---:|---|
|actual_unit_raw_recovery|0.967059|[0.9077690751205721, 1.026349552448883]|
|actual_unit_native_KL_recovery|0.711173|[0.6518832390111051, 0.7704637163394159]|
|actual_unit_remaining_effect_fraction|0.313061|[0.25377108015376343, 0.3723515574820743]|
|intact_next_token_NLL_change|0.446861|[0.3875702932659761, 0.5061507705942869]|

|Set|Family|Amplitude|Raw response fidelity|Native KL recovery|Remaining source-effect fraction|
|---|---|---:|---:|---:|---:|
|contrast64|actual|1|0.967059|0.711173|0.313061|
|contrast64|actual|4|0.974964|0.722986|0.289689|
|contrast64|rotated|1|0.463610|0.211839|0.897000|
|contrast64|rotated|4|0.641029|0.303814|0.780399|
|energy64|actual|1|0.969498|0.712161|0.312251|
|energy64|actual|4|0.976033|0.720207|0.285609|
|energy64|rotated|1|0.480107|0.214825|0.887779|
|energy64|rotated|4|0.660131|0.316717|0.758568|

|Set|Intact-output KL|Intact NLL increase|
|---|---:|---:|
|contrast64|0.401055|0.446861|
|energy64|0.389593|0.436511|
|random64_0|0.000648|0.001081|
|random64_1|0.001355|0.002953|
|random64_2|0.002029|0.005270|
|random64_3|0.001069|0.003824|
|random64_4|0.004505|0.004999|
|random64_5|0.000832|0.001696|

Audits: {'max_deleted_forward_normalized_state_absolute_error': 1.2665987014770508e-05, 'geometry_errors': [[3.7059116705784954e-08, 2.5198492483038342e-08], [3.2105418584506986e-08, 2.3374960892008702e-08], [3.015608031404468e-08, 2.078183411038097e-08], [3.346279681629923e-08, 2.3586058250878442e-08], [3.2920083237445176e-08, 2.2335747725003625e-08], [2.949764030810121e-08, 2.242501882824812e-08]]}. Independent arrays/documents: {'mean_kl_max_absolute_error': 5.551115123125783e-17, 'baseline_next_token_NLL': 3.2076060854336523, 'document_audit': {'excluded_rehashed_complete_documents': 7360, 'fresh_unique_documents': 512, 'fresh_unique_windows': 512, 'overlap': 0, 'completed_unique_total': 7872}}. The deleted-state algebra is checked against direct selected-factor subtraction at four sites in every source/amplitude case. All site scores, raw energies/errors, intact-output metrics and bootstrap arrays are saved. Scores retain numerical signs; native baseline effects are positive. No factor IDs or model weights are modified on disk.

Outcome: raw unit90 passes; native unit90, remaining unit20 and intact loss02 fail. Contrast64 actual unit raw-response fidelity is96.706%, joint interval[90.777%,102.635%]. The interval is untruncated and may exceed the mathematical upper limit because it is the frozen symmetric simultaneous band. Native-KL recovery is71.117%,[65.188%,77.046%]. Remaining head-effect fraction after factor removal is31.306%,[25.377%,37.235%]. Intact next-token NLL increases0.44686 nats,[0.38757,0.50615], from3.20761 to3.65447. All failed criteria remain failed.

At4× actual intervention, raw-response fidelity is97.496% but native-KL recovery is72.299%; remaining source-effect fraction is28.969%. Thus the main deficit is not failure of raw finite-response reconstruction at the larger tested amplitude. The native readout assigns importance to residual directions that the raw-energy ranking underweights. Energy64 performs similarly: actual unit raw96.950%, native71.216%, remaining31.225%, intact NLL increase0.43651 nats. This supports the earlier overlap result rather than a unique advantage of signed contrast ranking.

Source selectivity remains substantial: contrast64 rotated-control raw fidelity is46.361% at unit amplitude and64.103% at4×; native-KL recovery is21.184% and30.381%. Remaining rotated-control effect fractions are89.700% and78.040%. Six random64 sets average only2.464% actual unit native recovery and leave98.255% of the native source effect. These are descriptive comparisons outside the primary joint4 gate family.

The ordinary contrast-factor deletion has large baseline consequences: mean KL(intact||deleted intact) is0.40105, versus mean native actual unit source-effect KL0.002104. These are different KL targets/directions, so their ratio is not a formal mediation statistic, but their scales emphasize that factor removal is a much broader output change than the source edit. Random-set intact KL averages0.001740 and next-token NLL increase0.003304 nats. The selected factors are important to intact prediction, not merely to the experimental head6 path. Consequently remaining-effect reduction after ordinary deletion cannot be called a clean selective pathway knockout.

The new panel is independent of ranking and all previous experimental panels. Direct deleted-state versus factor-subtraction audit has maximum normalized-state absolute discrepancy1.27e-5; geometry checks pass the1e-6 construction threshold. No factor IDs were reselected and no readouts were fitted. These data confirm finite raw source-response support while narrowing the claim about native-output sufficiency and causal specificity.

Hypothesis update: sparse native support of head6 raw sensitivity extends to finite interventions, but raw-energy importance is not the same as output-distribution sufficiency. Broad baseline function and metric-dependent omitted directions are leading explanations. The selected factors are not dispensable nuisance units, and low factor count should not be interpreted as a harmless model compression or a semantic circuit.

Next high-information experiment: clamp the selected factors' edited product activations to their original intact values, rather than deleting their intact contributions. This yields state+r−rS and leaves the intact baseline exactly unchanged. Compare its remaining source effect with ordinary deletion and retained-response prediction under the same native readout. It tests mediation through the selected response changes without the large baseline shift. Keep the clamping intervention explicit; it is input-dependent activation patching, not a permanently pruned model. Then, if omitted native-output effects remain large, compare a ranking chosen in native output-probability geometry against raw-energy ranking on a separate discovery panel, with the same factor counts and held-out tests. Future fresh panels must exclude7,872 complete-document hashes from completed panels.
