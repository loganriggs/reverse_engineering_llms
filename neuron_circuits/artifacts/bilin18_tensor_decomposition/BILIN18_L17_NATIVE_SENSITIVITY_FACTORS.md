

## H405 — Native factors carrying the head6 sensitivity advantage

Earlier H324–H327 factor tests targeted compiler-residual responses with fitted or inherited readouts and found broad support. This experiment targets the newly confirmed native head6 first derivative, using the original L17 Down weights without refitting. Selection uses the older H400512-document panel; evaluation uses the disjoint H404512-document panel. Evaluation is held out from factor ranking, but has prior research use and is not a new experimental panel. Six joint-geometry-preserving rotations per panel are reused.

Write each first derivative as g=sum_j phi_j D_j. Signed attribution is phi_j<D_j,g>/S, where S is intact-final-state squared norm. Its sum equals||g||²/S up to arithmetic error; this attribution is not itself a single-factor removal effect. Rank by discovery actual-minus-mean-rotation signed attribution for head6. Control ranking uses mean(phi_j²||D_j||²/S) on actual discovery sources; six fixed random permutations provide nested random subsets. All orders and scores are saved before evaluation collection. Ranks16,64,256,1024 are fixed. IDs are zero-based native factors; no coefficient/readout refits occur.

For each retained subset, compute exact native factor sums for derivatives g0,g5,g6 and mixed coefficients M06,M56. Omitted output is full minus retained. These are interventions on the local MLP response map at a fixed recipient input; they are not a full forward-pass behavioral ablation, because removing factors also changes the natural MLP write and downstream state in an intact run. The common scale remains the intact original state. Record retained energy, omitted energy, signed cross energy and retained reconstruction error.

Frozen gates: {'factor_identity': True, 'contrast64_actual90': True, 'contrast64_remove80': True, 'contrast64_over_random10': True}. Contrast64 actual-g6 fidelity must have lower bound>.9; removing those factors must reduce actual-minus-rotated g6 energy advantage by>.8; reduction must exceed mean random64 by>.1. Four thousand paired evaluation-document/rotation-draw resamples, also resampling six random orders for the random contrast, yield one simultaneous joint3 band. Ranking is fixed; intervals do not include discovery-selection uncertainty.

|Primary measure|Estimate|Joint95% interval|
|---|---:|---|
|contrast64_g6_actual_recovery|0.959721|[0.9307008825107426, 0.9887413046772562]|
|contrast64_g6_advantage_reduction|1.018996|[0.9899762319541262, 1.0480166541206397]|
|contrast64_advantage_reduction_minus_random|0.990672|[0.9616513493247932, 1.0196917714913067]|

|Factors|Contrast actual-g6 fidelity|Energy-ranked actual-g6 fidelity|Random mean actual-g6 fidelity|Contrast advantage reduction|
|---:|---:|---:|---:|---:|
|16|0.811163|0.825214|0.011244|0.873041|
|64|0.959721|0.968417|0.027608|1.018996|
|256|0.970890|0.983989|0.125823|1.024850|
|1024|0.974687|0.990264|0.399157|1.018847|

Audits: {'max_relative_native_factor_replay': 2.1525593183469027e-06, 'signed_energy_sum_relative_error': 2.39068028887913e-08}. Independent saved-array checks: {'retained_omitted_energy_identity_max_relative_error': 9.524593980883115e-08, 'discovery_evaluation_document_overlap': 0, 'top64_contrast_energy_overlap': 54}. Full derivative and mixed-factor sums replay native analytical formulas. Frozen orders, attribution scores, sitewise subset metrics and bootstrap samples are saved. The factor list is provided separately.

Outcome: all four frozen gates pass. Contrast-ranked64 factors retain95.972% of actual head6 first-derivative energy fidelity, joint interval[93.070%,98.874%]. Removing them reduces the actual-minus-rotation energy advantage by101.900%,[98.998%,104.802%]; values above100% mean the remaining actual energy is lower than remaining rotation energy, not an invalid clipping event. Reduction exceeds mean random64 by99.067 percentage points,[96.165,101.969]. Factor derivative/mixed response replay error is2.15e-6; signed attribution sums agree with energy to2.39e-8 relative error.

The64 factors are1.39% of the4608 native factors, but this is sparse support for a particular derivative target, not compression of the whole MLP. Sixteen contrast-ranked factors already retain81.116% actual-g6 fidelity. The64-factor subset retains only36.095% rotated-g6 fidelity, indicating source-orientation selectivity. Its mixed-response fidelities are68.315% for M06 and63.458% for M56; g0 and g5 fidelities are56.137% and48.694%. The sparse first-order advantage therefore does not imply equally sparse full coupling or universal response support.

Ordinary individual-contribution energy ranking is slightly better descriptively:96.842% actual-g6 fidelity at64 factors and102.670% advantage reduction. Its top64 overlap54 of the contrast-ranked64. There is no evidence here that the signed advantage ranking is uniquely superior; both identify much of the same native support. Mean random64 actual-g6 fidelity is low, and all six random subset orders are retained rather than selecting a favorable seed. The strongest contrast-ranked native factor IDs begin351,2563,1082,1239,3533,1083,2908,2902; these are indices, not semantic labels.

Removing the contrast64 set also changes the descriptive primary curvature-to-sensitivity actual/rotation ratios from0.1373 to0.7294 for0/6 and0.1129 to0.7098 for5/6. Retaining only the64 gives0.1972 and0.1943. These subset ratio changes are descriptive, without additional frozen confidence gates, and are local response-map interventions. They connect the factor support to the earlier normalized-coupling pattern, but do not prove a downstream behavioral role or selective semantic function.

This revises the sparse-versus-distributed hypothesis: the head6 sensitivity advantage has a compact native-factor support that transfers between disjoint document panels, while mixed responses remain substantially broader under the same subset. Earlier poor factor-subset recovery for compiler residuals is not contradicted; the target, readout and ranking objective differ. No new readout was learned, and no global feature dictionary has been recovered.

Next high-information experiment: lock these factor IDs and test finite head6 interventions, not just local derivatives. Compare retained-factor finite-response prediction and factor removal at ordinary and larger amplitudes, with the native final normalization/readout and matched rotations. Separately quantify the intact-output change caused by removing the same factors, so an apparent intervention effect is not conflated with generic damage to the model. Use held-out/fresh documents and keep controls fixed. If finite predictions and targeted causal effects agree, inspect the factors’ activation/product patterns on controlled linguistic examples; semantic labels should follow, not precede, that evidence. The broad goal remains open.
