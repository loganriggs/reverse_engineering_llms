## H1034 — Whole-task transfer fails for the frozen sparse selectors

Previous turn was progress: published the current research snapshot and verified remote main. This turn inspects and independently audits the completed held-out-task evaluation; no training or candidate selection occurred.

Copying and spatial inversion were excluded entirely from the PCA and H1031–H1033 training. These are historically evaluated tasks, not fresh task discovery. All 32 equal-length pairs (64 endpoints) across four historical splits were retained. Only discovery and lexical cohorts in each family passed the prespecified native competence gate: at least 75% both-endpoint binary accuracy and 50% intended full-vocabulary top-answer accuracy. Structural and joint cohorts failed and remain diagnostics. Each cohort has only four pairs.

Results below average equally over the four competent family/split cohorts. Whole KL compares native output with reconstructed-state output. Paired relative KL compares a decoded donor-minus-recipient intervention with the full donor output, normalized by the no-edit divergence; 0 is perfect and 1 equals no edit. It is not a percentage of model understanding.

| Candidate | Whole KL | Paired relative KL |
|---|---:|---:|
| Scalar, untrained |4.2531|0.9156|
| Scalar, activation trained |4.0706|0.8867|
| Scalar, finite causal trained |4.2569|0.9093|
| Block, untrained |5.0317|0.9005|
| Block, activation trained |4.8969|0.8922|
| Block, finite causal trained |5.0343|0.9408|
| Shared, untrained |4.9277|0.9381|
| Shared, activation trained |5.1464|0.9140|
| Shared, finite causal trained |5.0486|0.9693|
| PCA16 per stream |5.1569|0.9698|
| PCA128 per stream |2.8755|0.7392|

Every candidate has zero intended full-vocabulary top-answer accuracy after whole-state reconstruction on these competent cohorts, versus native cohort accuracy 75–100%. Finite causal training worsens paired relative KL versus activation training by approximately 2.5% scalar, 5.4% block, and 6.1% shared. The limited development gains of H1033 do not transfer here. This does not establish that causal supervision is generally ineffective: the training set is small, only one seed is used, and the learned selector cannot change the fixed dictionary geometry.

The PCA128 reference uses 384 total coordinates, versus at most 48 for selectors, and is not a matched-budget competitor. Its poor transfer raises the possibility that important directions lie outside the discovery-trained span, or that simultaneous projection of multiple interacting streams is disruptive. Full-span Euclidean projection is not a behaviorally optimal encoder, so this is not proof that no representation in that span could work.

Audit: all source hashes unchanged, no training/evaluation prompt overlap, zero fitted evaluation rows, maximum 48 selected coordinates, 64 native prefix and 1472 suffix evaluations. Independently recomputed all 88 summary rows from records and KL from 11 saved endpoint logit samples, one per arm; maximum discrepancy 4.18e-14. This verifies metrics, not an independent rerun of model execution. Failed native cohorts were not silently dropped from the saved evidence.

Course correction: stop tuning these frozen selectors for now. Next run the eight native-versus-PCA128 stream combinations on the same frozen competent holdout panel, treating this as diagnosis rather than new validation. The three streams are the layer-11 residual, the initial embedding residual input, and cached first-layer values. Quantify which projected stream and interactions cause whole-output and paired-edit failure; measure retained state/difference energy alongside behavioral loss. Deliverable is a stream-specific representation requirement for the next joint featurizer comparison. Stop after this fixed factorial; do not tune on these holdouts. Subsequent training must retain whole-task exclusions and obtain a separate evaluation panel before claiming generalization.
