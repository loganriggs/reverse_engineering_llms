## H944 — Random-intervention training misses compact known relay responses

Evaluate all six frozen H943 models on the established H849 agreement/sentiment relay panel, which did not enter their training. All12 native competence cells pass. For each of156 directed matched-token/position endpoint contrasts, evaluate both full raw donor difference and projection onto the previously frozen own-task rank1 input direction:312 local intervention jobs. The known axes are evaluation-only. This is an existing research panel, not an untouched research-wide test.

The students miss nearly all MLP update changes. Across formats/gaps/seeds, explicit-effect model errors are96.9–99.7% for full agreement contrasts,96.6–100.6% for agreement rank1,97.6–98.2% for full sentiment and97.5–98.0% for sentiment rank1. Native/endpoint objectives are similarly poor and often slightly better. Thus the random-intervention training does not preserve these useful known mechanisms, even though the corresponding native tasks are competent.

Unlike H943's arbitrary random probes, the task-response output geometry is compact:

| Intervention family | Output dimensions for90% energy | Best rank1 error | Best rank8 error |
|---|---:|---:|---:|
| Agreement full contrast |11|64.3%|36.3%|
| Agreement rank1 input contrast |4|44.2%|17.8%|
| Sentiment full contrast |4|48.0%|17.0%|
| Sentiment rank1 input contrast |1|29.2%|10.3%|

These are evaluation-only SVD oracles, not learned predictors. Opposite-direction full donor effects are negatives and the panel contains finitely many lexical pairs, so exact rank64 fits are partly a sample-rank artifact; do not treat them as proof of global low rank. The low rank90 values indicate concentration within the evaluated panel, requiring prospective transfer checks.

Projecting the true effects into each effect-trained student's learned64-dimensional output span still leaves89.6–91.8% error. The low-dimensional task signal is mostly missing from that span. Therefore the global64-dimensional limit that obstructed arbitrary probes is not the whole explanation here: objective weighting, probe distribution, training coverage and optimization failed to select the relevant task directions. This is evidence against assuming random causal probes automatically discover meaningful units, not against causal objectives generally.

Local MLP effects are not the entire previously verified relay mechanism: changing rawpreMLP state also directly changes the residual stream. This experiment has not tested full-model output KL or the students' downstream answer effects. Those remain necessary before claiming task preservation or failure of the whole relay representation. We must separate the MLP-mediated part from the residual bypass when evaluating a trained repair.

Verification:624 local teacher evaluations, no whole-model forwards, no refitting or reserved-test evaluations. All model/data/direction hashes unchanged; all metrics finite; saved effect tensors reproduce every reported squared error exactly. Known task directions were absent fromH943 training. Natural-document pool unchanged.

Next: implement the user's known-mechanism anchor option explicitly. Train on the original discovery agreement/sentiment contexts or their independently defined source contrasts, alongside natural native/effect targets, and reserve H849 lexical/context variants for evaluation. Keep endpoint-augmentation controls and both seeds. Recovery of supplied mechanisms is supervised validation, not discovery; novel transfer, reusable conditional rules and additional held-out mechanisms are the discovery criteria. First measure the MLP-specific downstream effect with residual-path controls so the training target corresponds to computation that matters.
