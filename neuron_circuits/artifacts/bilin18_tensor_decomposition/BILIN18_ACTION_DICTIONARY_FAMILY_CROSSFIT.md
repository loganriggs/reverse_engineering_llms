## H1048 — Whole-family crossfit gains concentrate in country-based tasks

Previous goal turn was progress: H1047 established a continuous decoder recipe that genuinely lowers training loss and improves within-family edits. This turn ran the prespecified ten-fold whole-family exclusion test:30fits, scalar/block/shared for each excluded family. For each fold, PCA128perstream, center, scale, and decoder were fitted only on the other nine families. No exact held-out prompt entered fitted statistics.

Each method uses the same encoder input access,442368decoder parameters, fixed random group layout, max48active coordinates, and144fixed steps (16visits per training family). All remaining training pairs are covered. Optimizer/objective unchanged from H1047; initial and final checkpoints evaluated, with no outcome-based checkpoint choice. All30fits reduce their full-training equal-family KL, so this comparison is not undermined by the earlier STE optimization failure.

| Method | Mean training KL initial→final | Mean held-out relative KL initial→final | Held-out donor top1 initial→final |
|---|---:|---:|---:|
| Scalar |0.2841→0.1535|0.6485→0.6364|10.3%→13.1%|
| Block |0.8144→0.0912|0.7716→0.7322|0.0%→3.8%|
| Shared |1.1071→0.0887|0.8441→0.7963|2.5%→5.0%|

These are equal-fold macro averages. Block/shared obtain larger relative reductions from worse starting points but remain worse than scalar in final held-out relative KL in all10folds. This compares these fixed random support groups, not optimized semantic group memberships or arbitrary BSFs. One seed/partition and small panels limit broader architecture claims.

The averaged improvement is concentrated:

| Excluded family | Scalar initial→final | Block initial→final | Shared initial→final |
|---|---:|---:|---:|
| Capital recall |0.1624→0.1104|0.4015→0.2170|0.6768→0.3985|
| Country-language recall |0.1966→0.1296|0.4511→0.2534|0.5605→0.3225|
| Pronoun gender |0.4080→0.3696|0.6865→0.6381|0.7572→0.7518|
| Number successor |0.9235→0.9238|0.9539→0.9544|0.9709→0.9768|

Removing capital and country-language from the macro leaves essentially no aggregate gain: scalar0.76579→0.76554, block0.85794→0.85647, shared0.90045→0.90530. Agreement, antonym, binding, and several other excluded families remain poorly represented despite strong training fits. The method supports some task-to-task reuse, but not a broad transferable feature vocabulary.

Important interpretation constraint: the two factual families share country identities/words in the bank. Holding out capital recall still leaves country-language training on many of the same countries, and vice versa. This is a legitimate named-task exclusion but not a lexical/content or primitive exclusion. Earlier cached-value work already suggested shared country payloads across factual readers. These results are consistent with that mechanism, but the current experiment does not establish it or prove a general abstract factual-recall operation. Native errors are retained; fidelity reproduces the model, not an ideal answerer.

The bank was competence-screened historically. This crossfit enforces whole-family training exclusion for every learned statistic, but is not pristine external validation or new-task discovery. Decoder columns remain continuous action directions with a frozen PCA encoder and fixed support grouping; there is still no validated manifold/ontology or learned computational DAG.

Verification:18400suffix calls, including8640student training passes,10folds/30fits, all finite. Fold PCA orthogonality3.11e-15; max48active; exact held-out training prompt overlap0. Source hashes unchanged. Independently recomputed all60summary rows and KL for60saved held-out logit examples, max2.45e-14. Per-fold geometry, decoder/optimizer checkpoints, training/held-out records, and audit logits saved. Full training improvement confirmed for every fit.

Next focused transfer check: evaluate the frozen capital/language crossfit models on different country identities, using existing historical lexical panels or a fixed new prompt panel with competence checked separately. Preserve the trained models and compare seen-country versus excluded-country transfer without refitting or selecting ranks. The deliverable is whether the observed reusable action is tied to shared country identities or extends across content. Stop after that matched lexical-transfer test; do not reopen a lengthy head-level mechanism detour. Its result should guide the supervision split and shared-dictionary design for the next broader representation experiment.
