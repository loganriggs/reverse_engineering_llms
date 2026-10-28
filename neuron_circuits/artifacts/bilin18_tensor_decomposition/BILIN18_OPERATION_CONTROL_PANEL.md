## H653 — Diverse operations separate shared formatting from task-conditioned geometry

H652 repaired the malformed factual labels and preserved frozen predictor transfer. It left only copying and capital recall as substantive task families. H653 broadens to pluralization, antonyms, capital recall, month succession, addition, and subject–verb agreement. Each has eight explicit item records, crossed with three instructions: perform the task, copy the first input word/number, or always answer yes. Each instruction is crossed with Give/Provide and Answer:/Result:, yielding576 prompts. The first four items per family are discovery, last four transfer. Declarations, prompts, targets, splits, runner hashes and configuration are saved before inference. Single-target-token eligibility and explicit item pairing are asserted. All prompts are retained regardless of competence.

This is an exploratory diagnostic panel with familiar/reused vocabulary, not a blind new lexical replication. Controls share the input and answer suffix but differ in instruction length and meaning; those differences are part of what must be disentangled. Agreement copying always targets The, so its candidate score is degenerate; constant-yes candidate scores are also trivial and are not reported as evidence of competence. Addition copying targets the first number.

**Native competence is uneven.** Candidate accuracy selects from the family's target tokens; top1 is full-vocabulary next-token accuracy, not full generated-response accuracy. All task cells have0% exact next-token accuracy in these instruction prompts, so candidate success must not be called successful free generation.

| Family | Task candidate accuracy | Task top1 | Copy top1 | Constant-yes top1 |
|---|---:|---:|---:|---:|
|addition|12.5%|0.0%|81.2%|0.0%|
|agreement|50.0%|0.0%|100.0%|0.0%|
|antonym|93.8%|0.0%|75.0%|59.4%|
|capital|100.0%|0.0%|93.8%|0.0%|
|plural|100.0%|0.0%|81.2%|34.4%|
|successor|12.5%|0.0%|100.0%|0.0%|

These results qualify this panel as an assay of instruction-conditioned processing, with task competence measured separately. Low scores could reflect prompt/task alignment, priors or actual inability; this experiment does not distinguish them. Constant-answer prompts often fail too, so they are instruction controls, not guaranteed no-computation states.

**Held-out geometry distinguishes the task-specific instruction from its controls.** Concatenate normalized MLP input contexts across layers10–17. Compute context shifts for Give→Provide separately at each suffix, and Answer:→Result: separately at each verb. Average shifts over four discovery items. Predict each of four held-out task shifts using its own task mean, the same-family copy-control mean, the same-family constant-control mean, or the pooled mean of the other five task families. No held-out fitting or rescaling occurs. Error is Euclidean shift reconstruction error divided by the true shift norm; zero-change error1. Values average eight item/context cases per cell.

**Verb change:**

| Family | Own task | Copy control | Constant control | Other tasks |
|---|---:|---:|---:|---:|
|addition|0.195|0.715|0.779|0.685|
|agreement|0.222|0.721|0.725|0.817|
|antonym|0.395|0.751|0.756|0.661|
|capital|0.239|0.808|0.856|0.792|
|plural|0.276|0.612|0.651|0.723|
|successor|0.299|0.728|0.822|0.728|

**Suffix change:**

| Family | Own task | Copy control | Constant control | Other tasks |
|---|---:|---:|---:|---:|
|addition|0.260|0.496|0.456|0.622|
|agreement|0.252|0.642|0.555|0.588|
|antonym|0.487|0.663|0.558|0.561|
|capital|0.279|0.590|0.490|0.581|
|plural|0.323|0.431|0.489|0.587|
|successor|0.366|0.571|0.570|0.601|

Own-task means outperform all three alternative means in every family/axis cell. Control shifts nevertheless predict better than zero change in the displayed averages. Suffix changes are more transferable from copy/constant controls than verb changes in every family, suggesting a shared answer-format component alongside instruction-specific context. This is geometry, not a demonstrated output-relevant decomposition. The retained original model and supplied family identity remain essential. Task-specific instruction text and length could explain much of the separation without task execution.

Repeatable held-out shifts also appear in low-candidate-accuracy families. Unlike the withdrawn H651 result, the current declaration and target mapping are valid. However, this only establishes geometric regularity in poorly answered prompt families, not the previous frozen predictor's functional transfer across factual failures.

### Refreshed hypothesis tournament: twenty competing explanations

Ranks are qualitative priorities, not posterior probabilities. Several can coexist.

| # | Hypothesis | Current evidence / distinguishing test |
|---|---|---|
|1|Shared formatting plus instruction-conditioned processing|Leading description: controls partially transfer, own-task shifts transfer better. Test composed output effects for both components.|
|2|Lexical instruction identity drives much of the separation|Plausible: task words and lengths differ. Compare multiple synonymous instructions within each operation.|
|3|Answer-introducing suffix has a broadly shared component|Supported geometrically. Transfer suffix shifts across tasks and measure output fidelity.|
|4|Stable mean change plus small item-dependent correction|Supported in prior copying/capital functional tests. Test discovery-only correction in these new operations.|
|5|Upstream state predicts the useful correction|Established within earlier two families. Freeze a predictor for new operation panels.|
|6|Internal regularity can persist without task success|Supported geometrically in low-scoring families. Measure functional effects and actual continuations before a behavioral claim.|
|7|Target vocabulary or answer distribution drives separation|Open. Use different operations with shared output tokens and balance answer frequencies.|
|8|Input lexical identity drives correction coefficients|Open. Cross identical inputs with competing valid operations and hold out lexical groups.|
|9|Instruction length or position drives separation|Open confound. Add length/position-matched wording controls.|
|10|Residual scale and normalization dominate differences|Earlier normalization-only tests weaker, not excluded here. Compare context versus normalization replacements.|
|11|Distributed context changes compose nonlinearly|Supported in prior broad wording shifts. Repeat finite-degree composition tests across new tasks.|
|12|Apparent higher-order structure mainly reflects shift magnitude|Prior scaling experiments partly support. Compare norm-matched shifts across task/control contrasts.|
|13|A few shared operator directions suffice across tasks|Unproven. Fit pooled discovery basis, evaluate held-out tasks and probes.|
|14|Each task needs a distinct operator subspace|Possible; geometric separation alone insufficient. Compare pooled versus per-task bases at equal dimension.|
|15|Small discovery sets hide transferable structure|Prior expansion improved rank3. Expand independent items before increasing rank.|
|16|Rare item-specific directions dominate remaining output error|Open. Compare average versus tail errors and stratify by item without dropping failures.|
|17|Downstream readout makes similar internal changes functionally different|Prior endpoint dependence supports. Compare internal fidelity, centered-logit error and KL separately.|
|18|Attention rather than MLP context carries task-specific differences|Not excluded; present assay records MLP contexts. Cross attention/context replacements.|
|19|Local derivative fidelity fails at finite intervention sizes|Open. Sweep intervention magnitude and compare derivative predictions to actual outputs.|
|20|No unique global feature system exists, but distribution-relative equivalence classes are useful|Consistent with task/context dependence, not established. Compare compression under internal, output-KL and task-loss objectives with independent evaluation.|

The next priority is hypothesis1 versus2/3/7/9: does task-specific geometry buy output fidelity beyond shared formatting, or mainly track wording? Use this frozen panel to compare exact task context shifts with own-task, same-input copy, constant, and other-task discovery means under identical receiver operators and probe directions. Report every family, with competence and output fidelity separate. This functional test can reject a geometrically attractive explanation before investing in more elaborate compression. Synonym and length controls are needed afterward to distinguish lexical wording from operation identity. No family should be removed merely for poor scores.

Audits:576 whole-model forwards,576 explicit prompts, all targets single tokens, item pairing checked, finite logits, manifest/declaration/runner hashes unchanged. Context audit produces384 held-out prediction records; inputs and audit source hashed. No tangent rollout or output-effect prediction was performed in H653. No new natural-text documents; FineWeb remains26,112. Broad research goal active and incomplete.
