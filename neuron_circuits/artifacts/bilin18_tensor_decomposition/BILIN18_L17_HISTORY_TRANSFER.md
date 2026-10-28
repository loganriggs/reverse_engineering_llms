## H530 — Same-target history control preserves the compression trend

Use every H527 document's token128 as the same target. Histories contain tokens ending at127 with lengths32/64/96/128. Thus target identity and final input token are fixed within document. All H522 pooled3,072-source PCA/Fisher maps and reconstruction means remain frozen. The collector supports variable sequence lengths, so actual unpadded histories are used; no padding invariance assumption is needed.

| History length | Native CE | Mean-removal KL | Fisher36 recovery | Fisher144 recovery | Fisher576 recovery |
|---|---:|---:|---:|---:|---:|
|32|3.218244|0.0110757|75.90%|89.97%|99.01%|
|64|3.026162|0.0119629|73.82%|89.28%|98.95%|
|96|2.950959|0.0121820|72.98%|88.92%|98.93%|
|128|2.923721|0.0122229|72.40%|88.76%|98.90%|

Native observed-token CE decreases monotonically from3.21824 to2.92372 nats with longer available history, a mean reduction0.29452 on the same targets. Meanwhile Fisher36 replacement KL increases from0.002670 to0.003373 and normalized recovery falls from75.90% to72.40%. Fisher144 recovery decreases89.97% to88.76%, and Fisher576 decreases99.01% to98.90%. PCA shows the same qualitative trend. The mean-source-removal denominator also changes, from0.011076 to0.012223; both absolute KL and normalized recovery are therefore reported.

This rules out changing target-token identity as the sole explanation of H529's trend within these documents. History length, sequence start and effective position still covary; the experiment does not isolate a pure positional mechanism. The frozen maps remain useful across lengths, with a broad behavioral rank spectrum. No fit or selection occurred.

| History length | Map | Prediction KL | Raw recovery | CE change ± SE |
|---|---|---:|---:|---:|
|32|PCA36|0.002836247|84.50%|+0.004340 ± 0.001797|
|32|Fisher36|0.002669584|83.39%|+0.003484 ± 0.001739|
|32|PCA144|0.001278039|92.57%|+0.001808 ± 0.001176|
|32|Fisher144|0.001110892|91.69%|+0.001536 ± 0.001141|
|32|PCA576|0.000142602|99.04%|+0.000485 ± 0.000367|
|32|Fisher576|0.000109264|98.82%|+0.000769 ± 0.000328|
|64|PCA36|0.003366861|82.86%|+0.004297 ± 0.002005|
|64|Fisher36|0.003132193|81.64%|+0.003154 ± 0.001945|
|64|PCA144|0.001456331|91.94%|+0.001715 ± 0.001275|
|64|Fisher144|0.001282393|90.99%|+0.001397 ± 0.001240|
|64|PCA576|0.000161169|98.96%|+0.000283 ± 0.000388|
|64|Fisher576|0.000125946|98.73%|+0.000330 ± 0.000339|
|96|PCA36|0.003553449|82.58%|+0.005741 ± 0.002105|
|96|Fisher36|0.003292102|81.34%|+0.004347 ± 0.002061|
|96|PCA144|0.001525281|91.85%|+0.002483 ± 0.001320|
|96|Fisher144|0.001349776|90.88%|+0.002301 ± 0.001299|
|96|PCA576|0.000168477|98.95%|+0.000251 ± 0.000398|
|96|Fisher576|0.000130675|98.72%|+0.000303 ± 0.000353|
|128|PCA36|0.003673203|82.42%|+0.005318 ± 0.002136|
|128|Fisher36|0.003373318|81.20%|+0.003885 ± 0.002079|
|128|PCA144|0.001554504|91.78%|+0.002098 ± 0.001324|
|128|Fisher144|0.001373373|90.79%|+0.001790 ± 0.001306|
|128|PCA576|0.000173037|98.93%|+0.000134 ± 0.000409|
|128|Fisher576|0.000134474|98.70%|+0.000241 ± 0.000365|

CE changes are relative to the native model under the corresponding history. Recovery is relative to mean-source replacement, not model accuracy or context-benefit retention. The native model still computes all upstream representations and the downstream MLP. Small CE changes therefore cannot show that these coordinates alone compute the benefit of longer history. Documents and targets are reused and paired; this is a diagnostic, not independent fresh confirmation or evidence of unique features.

Audits: all model/data hashes checked; each history contains exactly its declared number of input tokens, ends at the same token127, and excludes target128. No padding is supplied. Head-write sums replay, and the128-token pre-MLP states and every map's KL endpoint reproduce H529 position127 within the prescribed tolerances. All per-document errors, native CE and states are finite and archived with document/history axes. No new documents:26,112.

Next H531: isolate the context-dependent attention-write update. For each shorter history L in32/64/96, use the same document's delta=s_128−s_L. Hold the native128-token pre-MLP receiver fixed, and replace its summed attention write by s_L+A(delta), where A is each frozen PCA/Fisher map at36/144/576. Equivalently, feed p_128−(delta−A(delta)) through the full residual-plus-MLP path. The zero-map control uses p_128−delta; the full-identity control must exactly reproduce native predictions. Report absolute prediction KL, recovery relative to removing the entire delta, raw delta reconstruction, and descriptive CE. Do not use the old reconstruction mean: it cancels in a difference. This targets the added-context update through L17 attention only, not the total effect of additional history, which also changes earlier residual computation and receiver state. The hybrid short-source/long-receiver intervention is artificial and must be labeled as such. No refit, subgroup/rank selection, or fresh-confirmation claim. It tests whether successful compression mostly preserves shared source activity while missing the context-specific update.
