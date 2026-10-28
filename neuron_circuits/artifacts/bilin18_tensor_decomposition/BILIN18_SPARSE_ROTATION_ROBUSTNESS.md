## H703 — Group subspaces survive fresh orthogonal teachers and noise; nonorthogonality breaks the pipeline

Keep H702 settings fixed and test three fresh random four-group structures covering eight atoms. Each gets paired clean,5%noise and nonorthogonal versions. Observation noise has isotropic coordinate SD equal to5%of training-clean RMS. Nonorthogonal atoms are normalized columns of Q(I+0.4cyclic shift), giving neighboring atom cosine about0.345. Train the same orthogonal sparse rotation, support-frequency group extraction and exact gate inference. One initialization per teacher, matched across conditions; held triples/all-group examples excluded from fitting and checkpoint selection. All nine runs use generating labels only for evaluation.

The new structures expose an important evaluation ambiguity: atoms with identical membership columns always coactivate. Their Gaussian coordinates permit rotations inside that coactivation block without changing the distribution. Two teachers also have only12distinct atom unions among16gate patterns, so exact generating gate labels are not always identifiable from the support. We therefore retain individual-atom/gate metrics but add group-subspace overlap and aggregate component-removal error for coactivation-equivalent blocks. These are posthoc evaluation refinements, not training changes.

| Teacher | Condition | Membership agreement | Mean group-subspace overlap | Held clean relative MSE, cost0.01 | Mean block-removal error, cost0.01 |
|---|---|---:|---:|---:|---:|
|7031|clean|100.00%|0.999995|0.000277684|0.000369224|
|7031|noise_5pct|100.00%|0.999946|0.000973489|0.00122476|
|7031|nonorthogonal|81.25%|0.753579|0.273465|0.365344|
|7032|clean|100.00%|0.999998|0.000108987|9.67575e-05|
|7032|noise_5pct|100.00%|0.999943|0.000796999|0.000901823|
|7032|nonorthogonal|81.25%|0.753586|0.253368|0.461967|
|7033|clean|100.00%|0.999997|0.000490699|0.000645739|
|7033|noise_5pct|100.00%|0.999950|0.00113833|0.00141721|
|7033|nonorthogonal|81.25%|0.748871|0.202038|0.358452|

All clean/noisy orthogonal runs recover exact group membership after alignment. Mean group-subspace overlap exceeds0.99999clean and0.99994noisy. Individual atom cosines0.914–0.960and large individual removal errors would misleadingly suggest failed recovery; aggregate coactivation-block removal errors are much smaller: mean9.68e-5–6.46e-4clean at the primary cost,0.000902–0.00142noisy. This supports subspaces, rather than arbitrarily chosen axes within them, as the recoverable units for these synthetic distributions.

At the smaller1e-5penalty, clean block-removal errors fall to5.96e-6–1.13e-5. With noise, exact support recovery can worsen (31.5–63.4%) while clean reconstruction remains good: retaining extra directions fits noise or small coordinate leakage. Thus tiny reconstruction error and exact-support correctness can diverge. Multiple group patterns with the same union explain some gate ambiguity, but not every support mismatch; basis estimation error and weak-coordinate decisions also matter.

The nonorthogonal cases fail materially: held clean relative MSE20.2–27.3%, membership agreement81.25%, mean group-subspace overlap0.749–0.754, and mean coactivation-block removal error0.358–0.462. This is not merely a labeling ambiguity. The orthogonal fitted dictionary is a structural limitation for overlapping nonorthogonal directions; the frequency-based group extractor also loses support patterns when coordinates mix.

The next specific test is training-data covariance whitening before sparse rotation, followed by unwhitening for reconstruction. That allows nonorthogonal decoder directions in the original coordinates while retaining an orthogonal search in whitened coordinates. It relies on additional distributional assumptions and must be checked on these controls rather than presumed valid. Keep original-coordinate reconstruction and block effects as evaluation targets. Only after such robustness checks should we consider a bounded early-layer pilot alongside BSFs; no broad LLM-manifold claim is established here.

Audits: source hashes unchanged, finite results, maximum fitted orthogonality error9.54e-7. Existing hyperparameters unchanged across conditions; clean-versus-observed error reported separately. Evaluation expanded after detecting coactivation/nonunique-union ambiguities, and the full run was repeated with those metrics. No original-model forwards/new documents; reserved LLM test split untouched. Broad goal active.
