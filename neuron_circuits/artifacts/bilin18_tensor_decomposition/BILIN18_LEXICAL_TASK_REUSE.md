## H1028 — Frozen country-content operator contributes to both language and capital tasks

Previous turn was progress: H1027 completed finite edge restoration with matched controls. This turn tests task reuse rather than continuing a language-only routing explanation.

Panel fixed before model evaluation: four original discovery country pairs (Italy/Japan, France/China, Germany/Spain, Russia/Greece), two endpoints, two language templates and two capital templates.32prompts total,16capital prompts. Capitals are Rome/Tokyo/Paris/Beijing/Berlin/Madrid/Moscow/Athens, each one GPT2token. Every endpoint pair changes exactly one source token. All templates place the target source immediately before final 'is', matching the original export's documented placement. Familiar vocabulary and opened discovery countries; this is task transfer, not new-country or sealed-test generalization.

Native both-endpoint binary accuracy100% in every template. Full-vocabulary top1: language_A87.5%,language_B87.5%,capital_A87.5%,capital_B100%. No outcome filtering. All pass the preregistered75%binary/50%top1 competence gate.

H1023 export hashes and weights remained frozen. Interventions compare opposite-country cache substitution in headslot0 at layer13 only; slot4 at the same layer as a predeclared alternative-head control; exported payload with exact recipient native routing; frozen language routing coefficients; and final-position-only native routing. Native routing remains an explicitly charged oracle, and native downstream computation turns writes into answer effects.

| Template | Cached13.0 aligned share of natural country-swap effect | Cached13.4 share | Frozen language-routing relative error against13.0path |
|---|---:|---:|---:|
| Language A |63.2%|−0.23%|2.8%|
| Language B |64.4%|−0.28%|4.1%|
| Capital A |58.9%|0.006%|16.1%|
| Capital B |54.3%|0.011%|11.7%|

Aligned share is dot(path margin changes,natural margin changes)/sum(natural margin changes²), not variance explained or fraction of model understanding. Separate path effects need not add under nonlinear downstream computation. Source-to-final native coefficient means are−0.6738/−0.6565(language) versus−0.5912/−0.6112(capital). Fixed language coefficients overshoot the capital-path effect. Their larger aligned full-effect shares (68.2%/60.5%) are not better mechanistic fidelity; error is evaluated against the actual isolated path.

Native-routing exported writes replay the cache intervention within2.86e-6absolute margin error; cached-value deltas match within1.47e-7relative. This numerical replay is expected from the native algebra and validates implementation. The additional empirical finding is the sizeable behavioral contribution on a second task. Slot4 is a native alternative-path control, not an equal-price learned featurizer comparison.

Interpretation: a reusable country-content transformation can support distinct downstream answers, while context controls routing strength and the live downstream model supplies task-specific interpretation. This favors sharing content/operation components across task groups rather than making language and capital features entirely separate. It does not prove one low-dimensional semantic manifold, identify the downstream task reader, learn the gate, or establish broad cross-domain reuse: both tasks are closely related country facts and only four country pairs are tested.

Audits:32prefix and224suffix sequences, all finite, export hashes unchanged, zero routingfitrows and zero reservedBSFtest examples. Saved prompts, all interventions, native outputs and routing coefficients. Natural-document pool26112unchanged. No active process. Broad objective incomplete.

Next after user resumes: return these operator-reuse findings to the multitask featurizer comparison. Distinguish reusable country payload coordinates from task-dependent response/gate coordinates; compare shared dictionary groups against task-specific groups with held-out combinations and matched budgets. Preserve native-operator reference and causal tests as controls; do not count native routing or suffix calls as learned predictions. User requested pause immediately after this experiment; no further experiment launched.
