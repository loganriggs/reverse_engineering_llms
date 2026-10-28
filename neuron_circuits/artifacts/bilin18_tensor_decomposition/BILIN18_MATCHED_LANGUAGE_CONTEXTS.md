## H913 — Exact surface matching succeeds, but the bare field prompts fail joint competence

The previous turn made progress: H912 found a reproducible high-rank language-minus-copy operator contrast, while identifying a major prompt-surface confound. H913 tests matched task/copy skeletons before drawing mechanistic conclusions.

For each eligible country, use E: “Country: Italy\nLanguage:” versus “Country: Italy\nCountry:”, and F: “Country = Italy; Language =” versus “Country = Italy; Country =”. Only the requested field token changes within each pair. Tokenization is checked before model evaluation.

### Matching and panel scope

All56 task/copy matched pairs have identical token lengths, identical final token IDs, and exactly one changed token. The panel contains112 rows from the same eight old and six additional eligible country pairs. The previously declared first four old pairs remain designated discovery and the remaining old/new pairs transfer; no operator is fitted here. Indonesia/Albania and Czech Republic/Slovakia again fail continuation-token eligibility and are excluded before evaluation. No correctness-based exclusions are made.

Old/new country identities have been used previously, but these prompt skeletons are new to the investigation. The earlier frozen context means remain hash-unchanged. No intervention or operator contrast outcome is measured in this experiment.

### Native competence

The prior gate requires at least four eligible pairs, both endpoint binary preferences correct for at least75% of pairs, and expected-answer full-vocabulary top1 at least50%. Both task and copy must pass for a competent matched comparison.

|Cohort|Format|Pairs|Task both-binary|Task top1|Copy both-binary|Copy top1|Joint pass|
|---|---|---:|---:|---:|---:|---:|---|
|old|E|8|1.000|0.562|1.000|0.312|False|
|old|F|8|1.000|0.562|0.875|0.000|False|
|new|E|6|0.833|0.333|1.000|0.500|False|
|new|F|6|0.833|0.417|1.000|0.167|False|

No cohort/format passes jointly. Old-country language reaches56.25% top1 in both formats, but copying reaches31.25%/0%. New-country copying reaches50% in E, but language only33.3%; F fails both. High binary preference therefore cannot justify a competent-task operator comparison.

### Prediction inspection and design implication

A fixed illustrative inspection (first endpoint of the first old/new pair in each condition) shows that copying often predicts a different country rather than merely an answer-format token. For “Country: Italy\nCountry:” the leading continuation is “ United,” with “ Italy” second. The semicolon/equality variant also favors “ United.” Thus the bare repeated field is not reliably interpreted as copying the supplied country. These examples are diagnostic, not an exhaustive explanation of every failure.

Do not compute a task-specific mechanism claim from this failed baseline panel. Next use a shared demonstration prefix that explicitly shows both requested outputs for the same input, with the prefix identical for language and country queries. For example an input record can contain both “Language: Spanish” and “Country: Argentina,” followed by a new input and either query. This can clarify the intended operation while preserving the one-token query contrast, equal lengths and common final punctuation. Fix demonstration choices and counts before evaluating competence, retain all failed conditions, and avoid separate task-specific prefixes that would reintroduce surface confounds. Demonstration countries should be outside the evaluation inventory.

Even if the revised prompts pass, the differing query word itself remains a semantic/lexical conditioning variable; identical final tokens do not isolate an abstract task independently of all text differences. The goal is a better controlled test, not a claim that all confounds vanish. Binding and broader feature/manifold work remain open.

### Verification

112 native full-model forwards. All56 matching checks pass exactly before evaluation. All model/source/frozen-mean hashes unchanged and margins finite. Eligibility manifest, full failures, native logits, L13/14 normalized captures, and illustrative top-five predictions saved. No natural or reserved-test evaluation; document pool remains26,112. Broad goal active.
