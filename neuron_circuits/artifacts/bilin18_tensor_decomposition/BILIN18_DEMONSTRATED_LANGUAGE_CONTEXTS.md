## H914 — Matched queries achieve joint competence, with a retained one-demonstration failure

The previous turn made progress: H913 verified exact token matching but found no jointly competent task/copy cohort. H914 tests a fixed common record schema and zero, one or two demonstrations, all declared before outcome measurement.

Query schema: “Input: Italy\nLanguage:” versus “Input: Italy\nCountry:”. N0 has no demonstration. N1 prepends “Input: Argentina\nLanguage: Spanish\nCountry: Argentina\n\n”. N2 additionally includes an Australia/English/Australia record in the same format. Demonstration order and countries are fixed, and Argentina/Australia are absent from evaluation. Each condition has exactly the same prefix for task and copying; only the final query field token differs.

N0 also changes the previous bare prompt's input label from Country to Input. Therefore its competence improvement is evidence about prompt design, not a demonstration effect. N1 versus N2 changes demonstration content as well as count; no pure count-effect claim is made.

### Matching and eligibility

All84 matched task/copy pairs have equal token lengths, identical final token IDs, and exactly one changed token. The same fourteen eligible country pairs yield168 rows. Indonesia/Albania and Czech Republic/Slovakia again fail single-token answer eligibility and are excluded before evaluation. All eligible endpoint results remain, regardless of correctness. Country identities have been used earlier; these record/query combinations are new to this investigation, not globally untouched data.

### Predefined split competence

The first four old pairs remain discovery; the next four old pairs are old-transfer; the six additional eligible pairs are new-transfer. The prior joint gate requires at least four pairs, both endpoint binary preferences correct for at least75% of pairs, and exact expected-answer full-vocabulary top1 at least50%, for both task and copy.

|Split|Condition|Pairs|Task both-binary|Task top1|Copy both-binary|Copy top1|Joint pass|
|---|---|---:|---:|---:|---:|---:|---|
|discovery|N0|4|1.000|0.500|1.000|0.875|True|
|discovery|N1|4|1.000|0.375|1.000|0.875|False|
|discovery|N2|4|1.000|0.875|1.000|1.000|True|
|old_transfer|N0|4|1.000|0.875|1.000|1.000|True|
|old_transfer|N1|4|1.000|0.250|1.000|1.000|False|
|old_transfer|N2|4|1.000|1.000|1.000|1.000|True|
|new_transfer|N0|6|0.833|0.500|1.000|0.917|True|
|new_transfer|N1|6|0.833|0.083|1.000|1.000|False|
|new_transfer|N2|6|0.833|0.500|1.000|1.000|True|

N0 and N2 pass jointly in all predefined splits. N2 old-transfer is100% top1 for both tasks. New-transfer language is only50% top1 in both passing conditions, exactly the threshold, and both-endpoint binary correctness83.3%; this is limited competence, not uniformly reliable performance. N1 language top1 falls to8.3% on new-transfer and25% on old-transfer despite high binary preference. The failure remains part of the experiment rather than being discarded.

A shared example prefix does not automatically improve a task. The one-demonstration degradation might reflect demonstration-answer bias or format effects, but this baseline screen does not establish the cause. No posthoc change to demonstration content, order or gate is made.

### Next controlled operator test

Use first-four discovery pairs only to derive context means separately for every condition/family, then freeze them. Evaluate full matched-source MLP changes and shared-partner approximations on both transfer splits at L13/14. Compare task and copying means on the same inputs, include original differently formatted maps as transfer controls where useful, and report all N0/N1/N2 outcomes. Interpret N0/N2 as gate-passing matched comparisons and N1 as a failed-gate diagnostic; do not call a condition selected by this competence screen fresh confirmation.

Compare the new task-minus-copy operator contrasts with the earlier surface-mismatched contrasts. This checks whether the recurring processing difference survives identical prefixes, final tokens and lengths. The changed query word still carries both lexical and semantic information, so even a positive result would not isolate an abstract task independently of its instruction token. Native competence alone provides no operator-fidelity evidence; interventions have not yet been run on this panel.

### Verification

168 native full-model forwards. All84 token-match audits pass exactly before evaluation; demonstration countries are explicitly disjoint from evaluation sources. All source/model/frozen-earlier-mean hashes unchanged and margins finite. Full eligibility/competence failures and L13/14 normalized captures saved. No natural or reserved-test evaluation; document pool remains26,112. Broad goal active; binding and wider shared-dictionary/manifold work remain incomplete.
