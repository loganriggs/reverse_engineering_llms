## H529 — Frozen source representations transfer across token positions

Freeze H522's pooled3,072-source PCA/Fisher maps at ranks36/144/576 and the shared reconstruction mean, all fitted at position63. Evaluate full residual-path replacements at positions31,63,95,127 on the same2,048 H527 documents. Positions are zero-based; labels are the next tokens32,64,96,128. Native inputs use the first128 tokens, and the final label is outside that input. Refit nothing.

| Position | Mean-removal KL | PCA36 recovery | Fisher36 recovery | PCA144 recovery | Fisher144 recovery | PCA576 recovery | Fisher576 recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
|31|0.0118743|75.37%|76.78%|89.41%|90.64%|98.79%|99.07%|
|63|0.0118801|71.17%|73.18%|87.97%|89.20%|98.57%|98.90%|
|95|0.0117736|69.48%|71.09%|87.36%|88.40%|98.55%|98.87%|
|127|0.0122229|69.95%|72.40%|87.28%|88.76%|98.58%|98.90%|

All frozen maps retain substantial fidelity at the additional positions. Fisher has lower mean KL than PCA in all12 rank/position comparisons. The broad rank spectrum persists: roughly71–77% recovery at36 dimensions,88–91% at144, and98.9–99.1% at576. This is evidence of transfer within this model/layer/context range, not proof of universal or unique computational units.

Rank36 error increases from position31 to later positions even though the mean-removal baseline stays around0.012. Fisher36 KL is0.002757 at31 versus0.003187/0.003404/0.003373 at63/95/127. Earlier-position recovery is consequently higher, rather than the calibration position being uniformly best. The experiment changes target token, available history and position together; it cannot identify whether this comes from context length, token composition, downstream sensitivity or source covariance. It does not isolate a pure positional effect.

| Position | Map | Mean prediction KL | Raw source recovery | CE change ± SE |
|---|---|---:|---:|---:|
|31|PCA36|0.002924374|83.42%|+0.004592 ± 0.001791|
|31|Fisher36|0.002757124|82.19%|+0.005022 ± 0.001759|
|31|PCA144|0.001257746|92.13%|+0.001413 ± 0.001122|
|31|Fisher144|0.001111438|91.19%|+0.001650 ± 0.001058|
|31|PCA576|0.000143276|98.98%|+0.000561 ± 0.000370|
|31|Fisher576|0.000110967|98.76%|+0.000355 ± 0.000326|
|63|PCA36|0.003425331|81.94%|+0.004386 ± 0.001864|
|63|Fisher36|0.003186769|80.67%|+0.003829 ± 0.001795|
|63|PCA144|0.001429581|91.46%|+0.001100 ± 0.001221|
|63|Fisher144|0.001283472|90.45%|+0.001086 ± 0.001163|
|63|PCA576|0.000170313|98.90%|+0.000147 ± 0.000412|
|63|Fisher576|0.000130406|98.66%|+0.000202 ± 0.000367|
|95|PCA36|0.003593030|81.42%|+0.004213 ± 0.001969|
|95|Fisher36|0.003403997|80.10%|+0.003721 ± 0.001929|
|95|PCA144|0.001487862|91.35%|+0.001149 ± 0.001216|
|95|Fisher144|0.001366096|90.32%|+0.001393 ± 0.001221|
|95|PCA576|0.000170751|98.89%|+0.000297 ± 0.000428|
|95|Fisher576|0.000132882|98.64%|+0.000218 ± 0.000373|
|127|PCA36|0.003673203|82.42%|+0.005318 ± 0.002136|
|127|Fisher36|0.003373317|81.20%|+0.003885 ± 0.002079|
|127|PCA144|0.001554504|91.78%|+0.002098 ± 0.001324|
|127|Fisher144|0.001373373|90.79%|+0.001790 ± 0.001306|
|127|PCA576|0.000173037|98.93%|+0.000134 ± 0.000409|
|127|Fisher576|0.000134474|98.70%|+0.000241 ± 0.000365|

Recovery is one minus replacement KL divided by the corresponding position's mean-source-replacement KL, not model accuracy or a fraction of total model computation. Raw source recovery is squared reconstruction-error reduction. CE is descriptive relative to each position's native predictions and is not a loss-neutrality claim. All positions share documents and are correlated. The documents were previously evaluated, so these results are diagnostic, not four independent fresh replications. No map/rank/position was selected.

Audits: all frozen inputs hash checked; attention head sums replay; position63 pre-MLP states match H527 within tolerance, and Fisher36 prediction KL replays within1e−4 relative tolerance. All per-document losses are finite. States, summed writes, target indices, raw error energies and paired KL/CE rows are archived with explicit document/position axes. No new documents:26,112 tracked. Native attention and MLP still run, so this is representation fidelity rather than a compute-saving implementation.

Next H530: hold the predicted target token fixed while varying preceding context. For each H527129-token window, use the same target token at index128 and histories ending at index127 of lengths32,64,96,128. Freeze all maps/means/ranks. Measure compression KL, native observed-token CE, replacement CE change and raw error at the last input position of each history. Short histories may be right-padded only for the collector; verify causal padding invariance against unpadded runs on a prescribed small subset before using that implementation. Replay the128-token endpoint against H529 position127. Keep document pairing and report every history length; do not select subgroups. This controls target identity and final input token while changing preceding evidence, although effective position and history length still covary. It discriminates a context-length dependence of the approximation from the target-token composition confound in H529. Reused-panel diagnostic, no new significance/uniqueness claim.
