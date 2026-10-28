

## H312 — Fresh neighbor-direction signal replicates; raw reconstruction fails

Frozen H311k16 rule, reference first768 discovery documents, new128 query documents excluding3648 prior complete-document hashes. Independent source/window/output hash and slice audit passes;128 unique new documents. Same head/payload and exact position matching. Context descriptor centers by reference mean only; normalized intervention delta; joint score averages the two cosines. Additive control averages the separate context and delta predictions without joint retrieval. No fresh-data fitting or selection. Cached reference response hash checked. Fresh angle seed20270206; neighbors seed20270207.

Two distinct prediction estimators are reported: direction prediction averages UNIT reference responses then normalizes; raw reconstruction averages RAW reference response vectors, which predicts amplitude without using query norms. Additive takes the mean of the respective context/delta vectors before scoring. The direction and raw metrics therefore do not isolate amplitude alone: unequal reference norms also change the predicted direction.

| Selector | Actual direction cosine | Angle direction cosine | Actual raw fidelity | Angle raw fidelity |
|---|---:|---:|---:|---:|
|random|0.0094|0.0034|-5.71%|-6.44%|
|context|0.1395|0.0458|0.79%|-2.36%|
|delta|0.1375|0.0277|-0.25%|-5.12%|
|joint|0.2132|0.1259|2.27%|-0.88%|
|additive|0.1752|0.0480|2.68%|-0.61%|

| Direction contrast | Family | Gain | Paired document95% |
|---|---|---:|---:|
|joint_random|actual|0.2038|[0.1971,0.2102]|
|joint_additive|actual|0.0380|[0.0361,0.0397]|
|joint_random|angle|0.1224|[0.1187,0.1262]|
|joint_additive|angle|0.0779|[0.0757,0.0800]|

Both directional gates PASS: joint-minus-random>=.05 and joint-minus-additive>=.02 in both families, positive paired95% lower bounds. Raw reconstruction>=5% in both families FAILS. Joint raw fidelity is2.27% actual (95%1.88–2.66%) and−.88% angle (−1.24 to−.53%). Negative fidelity means worse squared error than predicting zero missing response. Additive raw means yield2.68%/−.61%; no paired significance claim is made for additive-versus-joint raw ranking here. Bootstrap2000 resamples of128 documents preserves heads/sites; intervals condition on frozen references and one set of payload draws.

**Inference.** Joint geometric matching contributes beyond averaging separate context and intervention predictors, and this directional signal transfers to new documents from the cached corpus. This supports predictable relational geometry in the missing response, not merely reuse of validation outcomes. It does not demonstrate semantic units or an economical global computation: generic bilinear smoothness remains a viable explanation, absolute direction cosines are modest, and raw vector reconstruction is weak. The retrieval method stores a large native-supervised reference library and has not been compiled into a compact rule. No intact behavior or downstream KL test was performed.

**Audit.** All neighbor indices are in the3072 reference positions and match query position modulo4. Per-document saved errors exactly reproduce summary raw fidelities. Panel disjointness covers listed prior documents, not model-training data or external-distribution generalization. Native responses remain float32 with double squared-error accumulation. Original model/compiler/base unchanged; all jobs terminal.

**Next discriminator.** On existing discovery validation only, compare raw-neighbor means and unit-neighbor-direction predictions with reference-only amplitude estimates and fitted scalar shrinkage. Freeze scalar(s) using a distinct calibration subset, then confirm on a new panel if worthwhile. This separates poor scaling and norm-weighted direction changes from weak directional prediction; never use true query norms to claim executable fidelity. Also retain a simple bilinear geometry baseline before assigning semantic interpretation. Broad research goal remains active.
