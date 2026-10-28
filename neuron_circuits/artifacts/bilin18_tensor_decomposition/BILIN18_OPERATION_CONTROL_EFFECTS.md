## H654 — Task-conditioned context changes predict output effects beyond controls

H653 found held-out geometric regularity across six operation families with copy-first-token and constant-yes controls. H654 tests whether that task-conditioned advantage matters to outputs. It retains all six families: addition, agreement, antonyms, capital recall, plurals, and month succession. It uses four discovery items and four transfer items per family, and both Give→Provide and Answer:→Result: changes, each evaluated in two contexts of the other factor. This produces96 held-out contrasts.

Fit only the concatenated normalized MLP-context shift mean from the four discovery items, separately for family, control, axis, and setting. Compare own-task, same-input copying-control, same-input constant-control, and pooled other-task means. Exact context replacement and no replacement are reference conditions. The target context shift is added without renormalization across MLP10–17; receiver attention, MLP-normalization derivatives, and final readout remain fixed. No output fitting or parameter selection occurs.

Three seed654 orthonormal unit directions are injected directly into the final-position residual leaving block9. The same directions are used for every contrast and condition. This isolates the segment's operator differences; it differs from earlier block5 probes propagated to block9 and is not a replication of that exact probe distribution. The frozen mean/probe artifact and configuration are written before derivative evaluation. Native whole-block JVPs independently check the factorized native chain for every probe/contrast.

Error compares the predicted and exact context-replacement logit derivatives, divided by the magnitude of the exact context-replacement effect. The norm centers logits and weights by native receiver output probabilities. No-change error1; lower is better. This is local output sensitivity fidelity, not answer accuracy, finite-intervention KL, or a fraction of computation explained.

**Verb change, mean relative effect error:**

| Family | Own task | Copy control | Constant control | Other tasks |
|---|---:|---:|---:|---:|
|addition|0.250|0.763|0.901|0.841|
|agreement|0.239|0.781|0.794|0.915|
|antonym|0.403|0.797|0.792|0.738|
|capital|0.207|0.818|0.873|0.794|
|plural|0.295|0.658|0.710|0.905|
|successor|0.276|0.793|0.896|0.729|

**Suffix change, mean relative effect error:**

| Family | Own task | Copy control | Constant control | Other tasks |
|---|---:|---:|---:|---:|
|addition|0.281|0.611|0.467|0.612|
|agreement|0.238|0.605|0.518|0.570|
|antonym|0.456|0.615|0.511|0.532|
|capital|0.268|0.628|0.444|0.594|
|plural|0.293|0.396|0.483|0.575|
|successor|0.414|0.656|0.538|0.706|

Own-task prediction beats every alternative in all12 family/axis means. It beats copy in286/288 individual contrast/probe cases, constant in284/288, and other-task means in286/288. Every own-task case has error below1; its median error is0.279, maximum0.722. Cases share prompts, templates and probes and are not independent statistical trials. No case is excluded.

The exact effect is not merely a vanishing denominator: its magnitude relative to the native response ranges from0.082 to0.793, median0.301. Context replacement is one conditional operator effect; this ratio is not a causal contribution to the full donor response or the task.

Copy and constant controls transfer better for suffix changes than verb changes in every family, replicating the geometric pattern functionally. Their average errors remain below1, consistent with a shared component, but the own-task means consistently predict better. This is evidence that instruction-conditioned differences matter to local output sensitivity, beyond the particular generic controls tested.

The results do not identify operation-semantic units. Task instructions have different words and lengths; the constant-answer instruction is not reliably obeyed; and exact task next-token accuracy was0% throughout H653 even when candidate selection was strong. Addition and successor candidate accuracy were12.5%, agreement50%, antonyms93.75%, capitals/plurals100%. Thus functionally relevant regularity exists even in poorly answered prompt families, but it could describe instruction processing rather than successful operation execution. That inference uses valid H653 declarations, not the withdrawn malformed H651 factual labels.

The strongest current account is a shared format-sensitive component plus an instruction-conditioned transformation. The next discriminating experiment should cross operation identity with alternative synonymous instruction phrasings and matched suffixes. Fit on one wording and discovery items, then predict a second wording and independent transfer items. Include copying and constant controls and account for token length. If same-operation transfer survives wording changes and exceeds other-operation controls, semantic-operation structure becomes more plausible. If it fails while within-wording transfer remains strong, lexical instruction structure is the better explanation. Measure native competence separately; do not select only easy tasks after observing results.

This remains a conditional data-based description, retaining the original model and native receiver states. It does not establish a unique canonical form, minimal computation unit, or end-to-end surrogate. Shared versus task-specific functional components should be tested before additional compression optimization.

Audits:768 cached attention-stage evaluations;1512 batched JVP calls; no new whole-model forwards. Three independent native whole-block derivative checks have maximum relative errors below8.60e-7. Reconstructed normalized MLP contexts match stored float32 values within4.34e-6. Source, runner, and locked mean/probe hashes unchanged; all errors finite. All96 contrasts and three probes retained. FineWeb remains26,112 unique documents. Previous goal turn was progress: broad panel and twenty-hypothesis refresh completed. Broad research goal remains active and incomplete.
