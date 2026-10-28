## H550 — Positional response is task-dependent; copying needs downstream computation

Test the previously defined H533 transfer split in binding, copying, agreement and sentiment, with fixed L14H4 and L8H3 heads. This is transfer across tasks from the H549 distractor finding, not a new corpus or newly held-out prompt set. Keep all token-valid pairs, excluding sentiment-11 solely for unequal token lengths: 31 pairs, 62 prompts. Select the last differing source token. All pairs are candidate-correct at both endpoints. Full-vocabulary intended-group top-1 is 100% binding, 56.25% copying/agreement, and 64.29% sentiment, satisfying the pre-existing competence thresholds on each family.

Sweep five virtual key positions (start, halfway toward start, native, halfway toward destination, destination), modifying only the selected source's persistent-bus edge at the final destination. This is not a physical text movement. All 620 coefficients and frozen-downstream predictions were saved before live interventions. The frozen prediction adds the edge change, scaled by subsequent residual multipliers, to the native final residual and evaluates the exact final normalization/readout. It freezes same-layer and later MLP responses and later attention responses.

| Task | Head | Nearer weakens coefficient | Prediction direction agreement | Mean absolute live margin effect | Prediction MAE |
|---|---|---:|---:|---:|---:|
|agreement|L14H4|12.5%|68.0% (25 comparisons)|0.00048|0.00049|
|agreement|L8H3|12.5%|58.3% (12 comparisons)|0.00078|0.00077|
|entity_binding|L14H4|0.0%|100.0% (64 comparisons)|0.44901|0.01252|
|entity_binding|L8H3|0.0%|100.0% (64 comparisons)|0.08962|0.02086|
|sentiment|L14H4|78.6%|79.6% (49 comparisons)|0.00364|0.00289|
|sentiment|L8H3|0.0%|53.2% (47 comparisons)|0.01085|0.01070|
|sequence_copy|L14H4|50.0%|100.0% (64 comparisons)|0.05608|0.00398|
|sequence_copy|L8H3|100.0%|42.2% (64 comparisons)|0.04521|0.09879|

The simple directional extrapolation from H549 fails: nearer weakens neither binding head on any of 16 endpoints. It weakens L8H3 copying on all 16 endpoints, but L14H4 copying on only half. Position-dependent rotation combines with task-dependent query/key content; this evidence does not support a universal monotone distance bias. Sources also differ by task, so task versus role/content is not independently isolated.

Frozen-readout predictions get all 64 nonnative binding directions right for each head and all 64 L14H4 copying directions right. But L8H3 copying agrees on only 27/64, with prediction error 0.099 nats exceeding the mean absolute live effect 0.045. A direct identity-reader account is insufficient for this head/task. Agreement and sentiment effects are much smaller; their direction scores should not be interpreted as equally strong mechanistic failures. Direction scores omit predicted/live effects <=1e−4; magnitude metrics retain every nonnative condition. Repeated positions and related endpoints are not independent samples.

Audits: native-position margin replay within 1.10e−5, fixed predictions/manifest/source hashes, all finite. 620 interventions on reused prompts. No new FineWeb documents; total 26,112. Broad goal active. Next H551 clamps downstream MLP versus attention writes to identify which responses cause the copying mismatch. Physical text-position controls remain outstanding; the unexpected prediction failure is more immediately discriminating.
