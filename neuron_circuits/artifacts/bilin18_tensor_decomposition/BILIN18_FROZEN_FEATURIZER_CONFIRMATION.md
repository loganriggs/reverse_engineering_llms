## H837 — Frozen scalar advantage confirms on new documents and attention-write probes

Confirm H836 without refitting or selecting a favorable seed. Select512 documents from the existing natural corpus pool, excluding all2560 H681 and1024 H824 document hashes AND their128-token window hashes. New windows are mutually unique. One hash-selected position per document. Lock all20 H836 featurizers, rank64 global/residual/weighted/routed-weighted bases, probe seeds and a downstream subset before evaluating responses. These are new early-featurizer panel occurrences, not globally untouched research/model-pretraining documents.

### Frozen confirmation

Capture pre-MLP state, attention write and native MLP output at blocks1/2. Evaluate eight new orthonormal random raw directions (seed837+layer) plus the recipient's native attention-output vector normalized to unit norm. The attention-write probe is state dependent, not a donor swap or a validated task direction. All methods get the same direction, token baseline access and explicit radius according to their previously frozen definition.

Relative local response RMSE:

| Site / probes | Scalar64, two seeds | Scalar48, two seeds | Block8, two seeds | Fixed shared, two seeds | Learned shared, two seeds | Routed weighted PCA64 |
|---|---|---|---|---|---|---:|
| block1 / random |.531,.531|.555,.555|.588,.588|.685,.686|.695,.693|.604|
| block1 / attention write |.524,.525|.545,.545|.576,.574|.678,.680|.694,.693|.605|
| block2 / random |.586,.588|.612,.613|.654,.656|.765,.765|.778,.778|.678|
| block2 / attention write |.548,.549|.574,.575|.617,.617|.760,.757|.765,.766|.646|

The scalar advantage over block/shared alternatives transfers to this frozen new-document evaluation for both seeds and both probe families. Scalar48 also beats block8 despite its lower active count. This weakens an explanation based solely on the original random probe set or one favorable optimization seed. It does not establish general superiority over every BSF/shared-group architecture, convergence, semantic features or good manifold recovery.

Global PCA64 errors are .726/.758 for random probes and .717/.702 for attention-write probes. Conditioned/routed bases improve those baselines, but the trained scalar models remain stronger here. PCA versus learned dictionaries differs greatly in storage and support flexibility; no matched-parameter conclusion is claimed. Signed scalar/block controls share trainable parameter counts, while shared models have additional gate/membership parameters and different actual support counts as documented in H836.

### Scope and coverage

406 validation occurrences have tokens seen in sampled training positions;106 are unseen. All are retained, with full per-stratum metrics in JSON. Featurizers use the same global-mean fallback for unseen tokens as in development; no new routing is introduced for the trained models after seeing results. The only routed model is the already frozen weighted-PCA rule. Both seeds remain separate.

Errors are still substantial: the best scalar predictions retain roughly.52–.59 relative RMSE. The target is the MLP's local RMS/bilinear derivative, not full-model prediction distributions. No accuracy, CE, semantic ontology, selective reuse or task-generalization claim follows. Attention-write directions broaden the probe family but are not a substitute for diverse competent tasks.

### Downstream test locked before outcomes

The configuration freezes document indices0–31 of the new hash-sorted panel, probe0 (random) and probe8 (native attention write), and signed amplitudes +/-0.1 times each recipient's pre-MLP RMS. Future interventions will keep the skip perturbation, compare predicted MLP responses with the natural finite MLP response and include zero-response and own-analytic-response controls. All models/seeds and both sites must remain unless a documented implementation/validity failure prevents evaluation. Do not select the most favorable documents, dose or model after this local result.

This lock makes the next full-model KL comparison concrete; it has NOT been run yet. Saved raw predicted states, directions, input tokens, pre-MLP states and native MLP outputs permit direct implementation.

### Verification

512 three-block prefix forwards, zero full-model/test forwards. All source/model/fit hashes unchanged; document and exact-window disjointness pass; trained shared support union remains exact. All metrics finite. Original reserved test untouched; no new corpus acquisition, natural-document pool26,112. Saved manifest records source rows and window hashes; full per-document/probe errors and frozen predictions are retained.

Next run the locked downstream KL panel. If the local scalar advantage survives, use it as a fidelity baseline while comparing block/shared representations for stable geometry and causal meaning; do not conclude that scalar coordinates are the true computational units. If it does not, prioritize output-sensitive evaluation rather than further optimizing this local derivative metric. Keep the eventual held-out combinations, selective shared-component reuse and diverse-task tests explicit.
