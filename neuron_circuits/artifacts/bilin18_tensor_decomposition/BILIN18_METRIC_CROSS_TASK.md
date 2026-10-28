## H600 — Cross-task testing limits the number-trained representation's scope

Freeze H598's 64-dimensional context basis and response-weighted coefficient map. Reuse the entire H581 panel: copying, capital recall, entity binding, sentiment and agreement; two wordings, eight pairs, four variants each, 320 prompts. These are previously explored prompts, not a fresh task benchmark. No fitting or per-item filtering occurs. All five template0 task cells and copying template1 pass competence; agreement, recall, binding and sentiment template1 fail and remain in the results.

For each recipient, the context difference dc is its same-answer nuisance variant's native MLP11 input minus its own. Compare three incoming Attention5 signals: frozen number direction, recipient-to-opposite-answer native attention-write direction, and one seeded random direction. Normalize all to norm 41.8172 and test both signs. Equal upstream norm does not imply equal downstream MLP11 perturbation e. The task direction is constructed from paired writes, not a learned transferable task axis. Same-answer nuisance changes are not proven semantically inert.

Capture 960 perturbed responses. For each, run five live downstream interventions: exact cross term T(dc,e), no cross term, Euclidean projection, frozen response metric, and target-informed response-space oracle. 4,800 intervention forwards. Each adds its context-switch correction to the actual perturbed MLP11 write; no-context replay is exact. Coefficients for the fixed map use only K dc; actual dc and e are still observed.

Errors below are relative to the full-switch versus unchanged-context discrepancy. They are not task error rates or shares of total computation. P/M/O mean ordinary projection/frozen metric/per-case oracle. The oracle minimizes local response norm, not output KL or answer error.

| Task/template | Signal | Gate | Relative margin P/M/O | Relative KL P/M/O |
|---|---|---|---|---|
|agreement_0|number|pass|1.380/0.276/0.189|0.910/0.748/0.621|
|agreement_0|task|pass|1.452/0.269/0.382|0.900/0.788/0.659|
|agreement_0|random|pass|1.023/1.252/0.950|0.727/0.796/0.625|
|agreement_1|number|fail|1.532/0.496/0.296|0.915/0.789/0.682|
|agreement_1|task|fail|1.502/0.548/0.521|0.896/0.807/0.691|
|agreement_1|random|fail|0.878/1.127/0.813|0.910/0.964/0.792|
|capital_recall_0|number|pass|0.819/0.759/0.753|0.822/0.790/0.681|
|capital_recall_0|task|pass|1.210/1.281/1.381|0.808/0.798/0.621|
|capital_recall_0|random|pass|0.854/0.860/0.826|0.868/0.961/0.816|
|capital_recall_1|number|fail|0.562/0.481/0.575|0.649/0.668/0.539|
|capital_recall_1|task|fail|0.873/0.903/0.701|0.767/0.742/0.610|
|capital_recall_1|random|fail|0.782/0.856/0.906|0.844/0.845/0.767|
|entity_binding_0|number|pass|1.411/1.310/1.950|0.930/0.933/0.809|
|entity_binding_0|task|pass|0.862/0.864/0.955|0.899/0.937/0.723|
|entity_binding_0|random|pass|0.833/0.803/0.725|0.804/0.774/0.762|
|entity_binding_1|number|fail|0.840/0.860/0.553|0.786/0.792/0.581|
|entity_binding_1|task|fail|1.051/1.024/0.919|0.900/0.928/0.681|
|entity_binding_1|random|fail|0.772/0.919/0.941|0.800/0.749/0.729|
|sentiment_0|number|pass|1.062/0.886/0.712|0.830/0.725/0.757|
|sentiment_0|task|pass|0.524/0.460/0.444|0.917/0.948/0.814|
|sentiment_0|random|pass|1.895/1.748/2.537|0.860/0.858/0.755|
|sentiment_1|number|fail|1.276/1.547/1.696|0.872/0.861/0.906|
|sentiment_1|task|fail|0.791/0.779/0.619|0.774/0.808/0.737|
|sentiment_1|random|fail|1.157/0.883/1.205|0.840/0.922/0.798|
|sequence_copy_0|number|pass|1.000/0.985/0.925|0.868/0.884/0.759|
|sequence_copy_0|task|pass|1.068/0.958/0.882|0.936/0.895/0.797|
|sequence_copy_0|random|pass|1.270/1.269/1.072|1.042/1.063/0.862|
|sequence_copy_1|number|pass|0.917/0.873/0.685|0.825/0.756/0.550|
|sequence_copy_1|task|pass|0.929/0.871/0.780|0.743/0.691/0.588|
|sequence_copy_1|random|pass|0.944/0.819/0.859|0.857/0.709/0.703|

Among the six competence-passing task/template cells, both margin and KL improve in 4/6 for number, 3/6 for task, 3/6 for random. These counts describe this panel, not independent statistical replications.

The clearest preserved benefit is agreement's answer contrast: with its task-derived signal, relative margin error falls 1.452 to 0.269. Yet relative KL remains 0.788 and target-token CE fidelity barely changes (0.966 to 0.967). With a random signal, agreement's relative margin error worsens 1.023 to 1.252. For competent copying template0, task-signal margin improves only 1.068 to 0.958, and KL 0.936 to 0.895. These are weak approximations despite improvements. Sentiment task-signal margin improves 0.524 to 0.460 while KL worsens 0.917 to 0.948 and relative target-token CE error worsens 1.323 to 1.785.

Single-token CE is measured against the explicit intended answer token, separately from semantic-group NLL. CE fidelity here means preserving the full intervention's CE; it does not mean lowering the model's task loss. In failed-competence cells, intended answers remain labels, not proof of a behavior the model performs.

Most important, even the per-case oracle retains large local response errors: approximately 0.806–0.910 across all cells. The frozen context span lacks much of the new responses. Since the oracle uses the target response, changing only the fixed coefficient rule cannot remove that internal error within this span. This is not a lower bound on output KL or CE, nor a proof that the full computation lacks a compact representation in a different span.

Updated hypotheses and discriminating next experiment:

1. The agreement-trained context span has narrow coverage. Strongly supported by large oracle residuals. Broaden context discovery while holding rank fixed, then test held-out items.
2. The coefficient metric depends on the incoming signal distribution. Supported by agreement's number/task versus random contrast; one random direction is insufficient to characterize all generic directions. Compare number-only and multi-signal metrics in the same span.
3. A reusable representation exists but needs both broader context coverage and a broader response metric. Still open. Use a factorial of old/broad basis and old/broad metric to separate them.
4. Output-selective insensitivity hides large internal errors. Consistent with agreement margin improvement alongside weak KL/CE fidelity. Keep all three objectives separate rather than treating margin as complete faithfulness.
5. Prompt format or the kind of nuisance change drives some differences. Unresolved because context changes differ by task. Hold out wording and report each context-change type rather than pooling them as equivalent.
6. Each task needs unrelated units. Not established: a narrow span failing transfer cannot establish that no shared span exists. Compare a shared fixed-rank basis against task-specific bases with matched capacity and held-out evaluation.

Next use discovery items only to fit broader context coverage and multi-signal metrics, freeze before evaluating held-out items, and compare the factorial against the current map and oracle. Preserve failed competence cells and explicitly distinguish exploratory familiar-wording evaluation from fresh confirmation. This has higher information value than more agreement wording variants.

Audits: frozen sources/metric/bases and manifest unchanged; no refit; equal signal norms within 1.6e-5; exact native logit replay; oracle normal equations <=2.29e-15. A float32 four-term finite-difference audit initially shows up to 0.00150 relative cancellation error. Independent float64 evaluation on all 960 cases matches the saved float32 cross terms within 4.14e-7 and verifies the algebraic identity within 6.42e-13. No evidence of an intervention implementation error. Finite KL/CE/margin/internal metrics. FineWeb document count remains 26,112; broad goal remains active and incomplete.
