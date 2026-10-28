

## H407 — Clamp response changes while preserving the intact computation

Reuse H406512 documents, source/rotation draws, amplitudes1 and4, and all frozen factor sets. This is a paired diagnostic continuation, not fresh confirmation. For each edited input, replace selected bilinear product activations with their own intact values. If native edited MLP change is r and selected-factor change is rS, the clamped raw final state is s+r−rS. At zero edit it is exactly s. This is input-dependent activation patching, not a permanently pruned model. The direct residual remains fixed as in the MLP-path source intervention.

Score remaining effect KL(clamped edited||intact), prediction discrepancy KL(native edited||clamped edited), and clamped-edited next-token NLL minus intact NLL through exact final RMS/native unembedding/logit cap/softmax. Compare remaining-effect KL against H406 deletion KL(deleted edited||deleted baseline), using the same native source-effect denominator. These KL ratios quantify intervention effects; they are not an additive or uniquely defined percentage mediation decomposition.

Frozen gates: {'baseline_identity': True, 'native_replay': True, 'remaining_unit20': False, 'over_random10': True, 'deletion_equivalence05': False}. For contrast64 actual unit amplitude, remaining-effect fraction must have joint upper<.2; mean random remaining fraction minus contrast fraction must have lower>.1; clamp-minus-deletion difference must have its interval wholly within[−.05,.05]. Four thousand paired recipient-document bootstrap samples resample the six frozen random orders for the random contrast and use one joint3 band. Native effects must replay within1e-10; the no-edit clamp must preserve the intact state exactly. Baseline preservation is a construction property, not evidence of a harmless edited intervention.

|Primary statistic|Estimate|Joint95% interval|
|---|---:|---|
|actual_unit_clamped_remaining_fraction|0.290479|[0.2501620662161017, 0.33079616386443056]|
|random_minus_contrast_remaining_fraction|0.683899|[0.6435820351387735, 0.7242161327871024]|
|clamp_minus_deletion_remaining_fraction|-0.022582|[-0.06289925260181717, 0.017734845046511682]|

|Set|Family|Amplitude|Clamped remaining fraction|Deleted remaining fraction|Clamped NLL−intact NLL|
|---|---|---:|---:|---:|---:|
|contrast64|actual|1|0.290479|0.313061|0.000255|
|contrast64|actual|4|0.270579|0.289689|0.007700|
|contrast64|rotated|1|0.787564|0.897000|0.001389|
|contrast64|rotated|4|0.662425|0.780399|0.025001|
|energy64|actual|1|0.288965|0.312251|0.000272|
|energy64|actual|4|0.270978|0.285609|0.007789|
|energy64|rotated|1|0.784635|0.887779|0.001382|
|energy64|rotated|4|0.649938|0.758568|0.024471|

Audits: {'max_clamp_forward_state_absolute_error': 6.67572021484375e-06, 'baseline_state_absolute_error': 0.0, 'geometry_errors': [[3.7059116705784954e-08, 2.5198492483038342e-08], [3.2105418584506986e-08, 2.3374960892008702e-08], [3.015608031404468e-08, 2.078183411038097e-08], [3.346279681629923e-08, 2.3586058250878442e-08], [3.2920083237445176e-08, 2.2335747725003625e-08], [2.949764030810121e-08, 2.242501882824812e-08]], 'native_effect_replay_max_absolute': 0.0}. Independent saved-array checks: {'mean_score_max_absolute_error': 1.3877787807814457e-17, 'H406_native_effect_bitwise_replay': True}. Direct patching of product activations is compared with the state+r−rS expression at four fixed sites per source/amplitude case. Per-site KL and NLL changes, raw-response energies and bootstrap contrasts are retained. NLL differences may be negative and are not clipped. No new ranking, readout fitting or factor selection occurs.

Outcome: baseline identity and native-effect replay pass exactly; random-set advantage passes. The remaining-unit20 and deletion-equivalence05 gates fail. Contrast64 clamping leaves29.048% of the native unit source-effect KL, joint interval[25.016%,33.080%], versus mean random clamps97.438%. Random-minus-contrast remaining fraction is68.390 percentage points,[64.358,72.422]. The selected factor responses therefore carry a substantial source-specific output effect under a baseline-preserving patch, but are not sufficient under the frozen80%-removal criterion.

Ordinary deletion left31.306%; clamp-minus-deletion is−2.258 percentage points,[−6.290,+1.773]. Point estimates are close, but the interval is not wholly within the frozen[−5,+5] percentage-point equivalence range. Neither a large causal baseline effect nor equivalence is established by this contrast. The retained native-output deficit remains even when the large intact deletion effect is avoided. H406’s deletion-induced0.447-nat intact loss is not incurred at zero edit by this construction; baseline preservation is exact and expected from the patch definition.

At4× actual intervention, contrast64 leaves27.058% of native effect; rotated-control fractions are78.756% at unit and66.242% at4×. Energy64 is similar:28.897% and27.098% actual,78.463% and64.994% rotated. These are descriptive results beyond the primary actual-unit gate family. Clamped actual-source next-token NLL changes versus intact are+0.000255 nats at unit and+0.007700 at4×. They should not be compared as a formal performance-recovery fraction against an unreported native source-edit NLL change, nor interpreted as harmlessness for arbitrary inputs or interventions.

The full native-effect array replays H406 bitwise; no-edit clamping preserves the normalized state exactly. Direct patching of selected product activations versus the response-subtraction expression differs by at most6.68e-6 in normalized-state coordinates at audit sites. Frozen factor IDs, source interventions and all random sets are unchanged. The panel is reused and this is not an independent confirmation sample.

Hypothesis update: the sparse raw-sensitivity factor set mediates a large native-output effect under an activation patch, so the prior removal result was not solely a consequence of eliminating its intact write. However, roughly29% of the output-effect metric remains; raw-response energy rankings miss behaviorally relevant components. No semantic unit, complete circuit or universal compactness claim follows from the64-factor result. Native output geometry is now the leading practical discriminator for factor selection.

Next high-information experiment: rank factors on an older discovery panel using signed contributions to the head6 derivative in native probability geometry, including final RMS and logit-cap derivatives. This can be computed by pulling the probability-weighted output derivative back to the raw MLP response, without fitting a finite target. Freeze that ordering, then compare equal-size native-factor sets against H405 raw rankings and random subsets on held-out documents. Evaluate both retained-response native KL and baseline-preserving clamped remaining effect; preserve the90% and20% criteria. A native-metric ranking that improves held-out output fidelity would support measurement-specific factor importance, not a unique semantic ontology. If it fails, broader or interacting factor support remains a leading explanation. Future fresh panels still exclude7,872 complete-document hashes.
