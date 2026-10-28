## H1043 — Full pair exposure does not resolve expanded-selector failure

Previous goal turn was progress: H1042 completed the compute-matched joint pilot and identified incomplete direct causal exposure in the expanded80step schedule. This turn ran the fixed expanded160step control, keeping geometry, memberships, optimizer settings, pair order, and evaluation panels unchanged.

Because H1042saved model weights but not Adam state, first replayed steps0–79from initialization with the exact saved expanded PCA and same native teacher outputs. Every state-dictionary tensor matches the original80step checkpoint exactly for scalar, block, and shared (maxdifference0). Continued with the reconstructed Adam state through160steps, saving optimizer state at80and160. Thus this is a verified trajectory continuation, not a reset-optimizer fine-tune. No geometry refit and no evaluation-dependent stopping.

Expanded160has16visits per family, matching narrow80per-family exposure at twice optimization compute. All80expanded training pairs now receive finite-causal supervision, versus56at expanded80. This is not a convergence test and does not establish that more training could never help.

Paired relative KL, lower better:

| Panel / selector | Narrow80 | Expanded80 | Expanded160 |
|---|---:|---:|---:|
| Same-family / scalar |0.5265|0.7082|0.7000|
| Same-family / block |0.7750|0.8373|0.7991|
| Same-family / shared |0.8863|0.8844|0.9755|
| Historical exclusions / scalar |0.9125|0.9102|0.9026|
| Historical exclusions / block |0.9433|0.9447|0.9405|
| Historical exclusions / shared |0.9675|0.9628|0.9647|
| Weekday / scalar |0.8709|0.7743|0.8911|
| Weekday / block |0.9703|0.9190|0.9087|
| Weekday / shared |0.9601|0.9821|0.9417|

No consistent benefit across architectures and panels. The scalar weekday benefit at80steps disappears at160; the80step result must not be promoted as a reliable winner through retrospective checkpoint selection. Historical improvements are small and whole-reconstruction top-answer accuracy stays0for all candidates on copying/spatial and weekday panels. Same-family scalar macro top1recovers5.8%→10.8%, still below narrow32.5%. Shared same-family paired fidelity worsens despite slightly improved whole KL, again separating reconstruction and causal edit objectives.

All claims remain conditional on fixed training-only PCA128perport, fixed random groups, max48active coordinates, a diagonal importance selector, straight-through gradients, one seed, and opened small diagnostic panels. This does not falsify learned overlapping dictionaries, nonlinear manifolds, computational DAGs, or broader causal supervision. It weakens incomplete pair exposure as the sole explanation for H1042failure. Untrained geometry already produced much of the narrow/broad scalar tradeoff.

Verification:2548suffix calls this turn, including1920student passes (half reconstructing the original trajectory); three final fits, all80pairs covered, source and original checkpoint hashes unchanged. Max48/min41active coordinates in evaluation. No training/evaluation exact prompt overlap. Independently recomputed KL on9saved logits (one endpoint per panel/method), max4.18e-14. Corrected inherited metadata to state that geometry is reused, not refitted; no new PCA orthogonality test was claimed. Frozen PCA's earlier audit remains the source for geometry verification.

Stop further duration extensions for this selector family. Next fixed comparison targets the quantity being sparsified: current decoded-donor minus decoded-recipient activations versus a direct sparse encoding of the donor-minus-recipient difference. Both receive the same endpoint pair and use native recipient context, but the latter chooses support for the change itself and avoids changing endpoint supports. Compare at fixed48coordinate maximum and include dense128perport change projection as a capacity reference. This is an edit representation, not an autonomous whole-state featurizer or a learned semantic operation. Use frozen expanded geometry and no output-dependent support selection, fitting, or rank sweep. It tests whether context allocation/support switching, rather than solely task inventory, is limiting causal fidelity. Results should guide whether the next shared/block representation should encode changes conditionally on context.
