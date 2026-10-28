## H568 — Shared phrase-conditioned query edits transfer unevenly across readers

Fit mean raw-query changes using only the first four lexical pairs in H566 (eight endpoints per task/phrase/head). Differences are edited query minus base query, before per-head normalization and rotary position. Freeze means before any new-panel model evaluation. At inference add the corresponding mean change to the recipient's base query, then apply native normalization and the edited rotary position. Compare first-query-only and both-query updates against phase-only, base-coefficient, exact and deletion controls. No coefficient or outcome fitting.

Evaluate Leonard/Sandra, Howard/Ruth, Norman/Deborah and Philip/Carol in binding and copying with the six existing insertions and base: 112 panel prompts, 224 gate predictions saved before edited-prompt evaluation, 1,344 interventions. This tests lexical transfer within fixed phrases/templates; it does not test unseen phrases or identify a general semantic edit feature. Individual vocabulary and some base constructions may have appeared in earlier unrelated work. The discovery and evaluation lexical pairs for this fit are disjoint.

Native copying is 8/8 candidate and full-vocabulary correct in every cell. Binding is 8/8 candidate in all but “as people said” (7/8), and 6/8 full-vocabulary throughout. With four pairs, that exceptional binding cell has 3/4 complete pairs correct; it still meets the existing gate. No failed endpoint is dropped.

| Task/head, pooled edited prompts | Phase-only coefficient MAE | Shared first-query MAE | Shared both-query MAE | Phase-only / first / both live-margin MAE |
|---|---:|---:|---:|---|
|binding/L8H3|0.01533|0.00732|0.00436|0.01601 / 0.00757 / 0.00457|
|binding/L14H4|0.26653|0.01680|0.01323|0.30119 / 0.01880 / 0.01519|
|copying/L8H3|0.02323|0.01166|0.00880|0.00531 / 0.00313 / 0.00231|
|copying/L14H4|0.02570|0.01975|0.01822|0.01697 / 0.01338 / 0.01229|

Shared first-query edits reduce binding L14H4 coefficient MAE by about 94%, with a comparable live-margin improvement. L8H3 improves about 52% for binding and 50% for copying. Copying L14H4 improves only 23%, despite the first factor being causally dominant in H567. This distinction matters: knowing which factor carries the correction does not imply its change is invariant across lexical contexts.

Both-query means improve all four pooled cells further, but leave residual error. These are pooled summaries across six phrases, not a guarantee of improvement in every prompt. The local intervention measures how much replacing the coefficient disturbs the actual edited output; it does not predict the entire output effect of changing the sentence. Other pathways remain native to the edited prompt.

The successful binding transfer suggests a reusable query adjustment across these names, expressed as an additive change in a model-defined query space. However, it is conditioned on task, phrase and head. We have not shown that phrase specificity is necessary, or that it represents the meaning of “as usual,” “in fact,” etc. A generic insertion response could explain much of the success.

Next H569 compares the locked phrase-specific edits against pooled same-task edits, wrong equal-length phrase edits, and same-phrase cross-task edits on the same panel. These are diagnostic controls, not new independent confirmation. They distinguish phrase-specific information from generic insertion, length and task-dependent response. A later prospective phrase/template panel is required before claiming a transferable linguistic feature.

Audits: locked query-edit file and predictions unchanged; source keys match within 4.58e−5; self margin replay <=1.86e−5; equal-length groups verified; manifest/source unchanged and all finite. H567–H568 complete 2,880 interventions. H568 introduces a 112-prompt prospective panel; no new FineWeb documents, tracked total 26,112. Broad goal active and incomplete.
