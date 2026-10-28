## H569 — Generic within-task insertion response explains much, but not all, query-edit transfer

Reuse the 96 edited H568 evaluation prompts and locked discovery query means. Compare both-query updates from: matching phrase/task; all six phrases pooled within task; the other two same-length phrases pooled (target phrase excluded); each of those wrong same-length phrases separately; and the matching phrase from the other task. Include exact and phase-only controls. No refitting or outcome selection. Two heads and eight modes give 1,536 interventions.

| Task/head | Phase MAE | Matched | All phrases pooled | Other same-length pooled | Other-task |
|---|---:|---:|---:|---:|---:|
|binding/L8H3|0.01533|0.00436|0.01089|0.01264|0.01362|
|binding/L14H4|0.26653|0.01323|0.03788|0.05768|0.25004|
|copying/L8H3|0.02323|0.00880|0.01425|0.01956|0.01814|
|copying/L14H4|0.02570|0.01822|0.02753|0.02986|0.10623|

Binding L14H4's matched edit removes about 95% of phase-only error, while the generic same-task pool removes about 86%. Therefore much of the successful transfer does not require knowing the exact phrase. Matching the phrase still helps: its error is roughly one third of the all-phrase pool. Cross-task transfer nearly loses the benefit there, and harms L14H4 copying relative to phase-only prediction. At L8H3, phrase-specific corrections help more reliably than several mismatched controls.

The same-length pool excludes the target phrase, so its remaining benefit cannot be attributed to retaining that phrase's mean. However, task labels also identify different templates, source roles and context structures. These results do not isolate an abstract task feature. The wrong controls use both alternatives in fixed phrase order; all results are retained.

Audits: matched-condition replay against H568 and self conditions remain within numerical error; donor/source hashes unchanged; all finite. Reused panel, no new prompts or corpus documents. Next H570 freezes the pooled edits and evaluates unseen phrases crossed with original and changed templates, without fitting any new query means.

Audit values: {"self_margin_max_error": 1.8596649169921875e-05, "matched_replay_max_error": 8.106231689453125e-06, "source_unchanged": true, "donors_unchanged": true, "finite": true}
