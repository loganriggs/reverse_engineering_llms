## H859 — Frozen norm calibration confirms average output fidelity gains, with individual and token-coverage limits

The fixed single-input calibration lowers mean next-token KL in all20 model/coefficient comparisons on512 documents disjoint from featurizer training and development. All20 paired95% bootstrap intervals exclude zero on the improvement side. However,37–49% of individual documents get worse, and the unseen-token stratum is mixed. This verifies a useful inference option, not a semantic manifold or universal improvement.

### Context and definitions

The featurizers approximate the model's1152-dimensional state immediately before the layer4 MLP. Ten fits comprise signed sparse scalar dictionaries at48/64 active coordinates, blocks of8, fixed overlapping groups over shared atoms, and learned overlapping groups; two seeds each. These are the existing finite-training-budget implementations, not a matched-parameter architecture verdict or an exact Grassmannian BSF reproduction.

For each input, subtract its training-derived token mean. The calibration restores the length of the bias-centered reconstructed residual to the native residual length, then restores the input's overall raw RMS. It carries one extra native norm scalar; it is not a free improvement at identical information budget. LS means least-squares coefficient fitting within the support already selected by the encoder; no gates or dictionaries are retrained. Learned and LS coefficients each have calibrated and uncalibrated versions.

KL(native||replacement) measures disagreement with the original model's entire next-token distribution; lower is more faithful. Ground-truth cross-entropy is recorded separately and answers a different question. Replace the full raw state at a locked position and run the rest of the model, using the prefix ending there. This is direct reconstruction replacement, not a donor-difference task intervention.

### Confirmation results

|Model|Learned KL|Learned + norm KL|LS KL|LS + norm KL|
|---|---:|---:|---:|---:|
|block8_8351|0.0737|0.0683|0.0672|0.0602|
|block8_8352|0.0788|0.0706|0.0716|0.0647|
|scalar48_8351|0.0573|0.0499|0.0548|0.0476|
|scalar48_8352|0.0614|0.0547|0.0577|0.0508|
|scalar64_8351|0.0502|0.0452|0.0475|0.0426|
|scalar64_8352|0.0504|0.0435|0.0473|0.0404|
|shared_fixed_8351|0.1831|0.1396|0.1457|0.1105|
|shared_fixed_8352|0.1718|0.1339|0.1423|0.1109|
|shared_learned_8351|0.2033|0.1592|0.1878|0.1425|
|shared_learned_8352|0.1908|0.1512|0.1769|0.1385|

All512 documents and all40 variants were retained with no gain tuning or seed selection. Bootstrap resamples pair the same documents before/after calibration,8192 draws. Intervals are pointwise and uncorrected for multiple comparisons; the20 comparisons share documents and are not independent replications.

The406 occurrences whose tokens appear in the training token-mean table show lower mean KL in all20 comparisons, with all20 pointwise intervals below zero. Among106 unseen-token occurrences, only10/20 mean comparisons improve and only3 intervals lie wholly below zero. Unseen tokens use the global mean. This association does not isolate token coverage from other distribution differences, and does not authorize a retrospectively selected coverage-dependent calibration rule without new validation.

Depending on fit and coefficient method,37.1–49.0% of documents have increased KL. Worst observed increases range from0.104 to1.637 nats across comparisons. The saved analysis includes95th-percentile harm, maximum harm, median KL, total variation, and next-token CE. Positive mean performance does not imply safe per-example replacement.

### What this changes

Combined with H857's worse activation reconstruction and local MLP derivative errors, the confirmation strengthens the case for measuring whole-model fidelity directly. Reconstruction error alone is insufficient for choosing representations for causal interpretation. Shared-group fits still have poorer absolute fidelity than the sparse scalar fits here; restoring norm does not fix collapsed membership coverage or establish reusable shared computation.

The manifold program remains the objective. Next, freeze calibrated and uncalibrated scalar/block controls and examine whether candidate components recur across seeds, support coherent within-component variation, and produce specific causal changes across competent tasks. Include agreement and sentiment relays plus binding/copy/capital tasks at their demonstrably active sites, rather than treating a weak early-layer effect as a failed feature. Predefine matched random-subspace, token-conditioned, and off-component controls. Held-out combinations and selective component removal are the tests for reuse; visual clusters or reconstruction gains alone are insufficient. Keep shared-group alternatives as explicit comparisons and diagnose coverage before giving their groups semantic names.

### Audits and scope

21,504 whole-model forwards; exact native-state replacement reproduced logits exactly. All source hashes and panel window hashes matched;512 document hashes are disjoint from all2560 original and1024 development documents. SVD orthogonality error5.22e-15, norm-retention error2.14e-14, no LS reconstruction-error increases, all metrics finite. No reserved-test evaluations. Natural-document pool remains26,112.

This panel is new to layer4 featurizer fitting/selection and norm-rule output evaluation but was used in earlier research at other layers. It is not an untouched global research holdout or known pretraining-disjoint data. Broad research goal remains active.
