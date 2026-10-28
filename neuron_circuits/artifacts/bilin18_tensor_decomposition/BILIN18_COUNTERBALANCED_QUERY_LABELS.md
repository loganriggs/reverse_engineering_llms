## H916 — Arbitrary-label remapping fails symmetrically competent language behavior; demonstration order is confounded

The previous turn made progress: H915 found query-specific L13 processing under matched surfaces, but the lexical query words still differed. H916 tests arbitrary labels A/B whose requested operations swap across contexts, before fitting or interpreting any operators.

Mapping0 assigns A=language and B=country copying. Mapping1 reverses that assignment. D0/D1 provide two shared demonstration records (Argentina/Spanish and Australia/English); I0/I1 add an explicit mapping instruction. Each record lists A then B. The same final query label can therefore request either operation across mappings. Demonstration countries are excluded from the evaluation inventory. All four conditions are fixed before outcomes, with every eligible result retained.

### Token controls and native competence

112 within-mapping task/copy pairs have identical lengths/final tokens and exactly one changed query token. Another112 cross-mapping checks verify the same query label/suffix and equal length when comparing different requested operations. These controls do not make the whole prefixes identical across mappings; their content necessarily specifies different assignments.

Fourteen eligible country pairs produce224 rows. The prior continuation-token exclusions remain unchanged. Competence uses the previously defined minimum-pair, both-endpoint binary, and50% exact top1 thresholds.

|Cohort|Condition|Task top1|Copy top1|Task both-binary|Copy both-binary|Joint pass|
|---|---|---:|---:|---:|---:|---|
|old|D0|0.000|1.000|1.000|1.000|False|
|old|D1|0.000|1.000|1.000|1.000|False|
|old|I0|0.875|1.000|1.000|1.000|True|
|old|I1|0.312|1.000|1.000|1.000|False|
|new|D0|0.000|1.000|0.833|1.000|False|
|new|D1|0.000|1.000|0.833|1.000|False|
|new|I0|0.417|1.000|0.833|1.000|False|
|new|I1|0.000|1.000|0.833|1.000|False|

Copying has100% top1 throughout. Demonstration-only language has0% top1 in both mappings. Adding explicit instructions improves mapping0 language, with87.5% old-country top1, but mapping1 reaches only31.25%. New-country instruction cells both fail (41.7%/0%). There is no jointly competent pair of counterbalanced mappings; old I0 alone cannot support a mapping-invariant semantic-operator claim.

### Binary source discrimination concealed wrong-task preference

The prior binary margin compares two countries' language answers, not language versus copying the current country. A diagnostic comparison of the intended language against that same country's name shows language is never preferred in D0/D1, in either cohort. Old I0 prefers language in93.75% of endpoints; old I1 only31.25%. New I0 reaches50%, new I1 zero. Thus high binary language preference can coexist with preference for the other operation's answer. This validates the need for full-vocabulary competence and suggests retaining a task-choice diagnostic in future screens.

These comparisons describe output preferences, not complete internal task selection or proof that no language computation occurred. No intervention or operator fit is run here.

### Design confound and next test

The record order was fixed A then B. Swapping label meanings therefore also swaps whether language output appears first or second in demonstrations; the explicit instruction's semantic ordering changes too. The observed asymmetry cannot be assigned solely to label identity or mapping ability.

Next cross label mapping with demonstration field order, and control instruction ordering separately or hold semantic instruction order fixed. For example, keep “language is field X; country is field Y” as the instruction sequence while X/Y swap, and evaluate both A-first and B-first records. Predetermine all combinations and retain failures. Check joint competence and same-country language-versus-copy preferences before interpreting operators. Do not rescue the present test by selecting only the successful mapping or dropping difficult countries.

This failed screen narrows the experimental design; it does not refute abstract task representations or validate a lexical-only account. The model may be sensitive to demonstration order or arbitrary-label instruction format. Binding pathways and the broader shared-dictionary/manifold objective remain open.

### Verification

224 native full-model forwards. All112 within-mapping and112 cross-mapping token-control checks pass. Demonstration/evaluation countries are disjoint. Source/model/frozen-earlier-mean hashes unchanged and all margins finite. Eligibility failures, native logits, normalized L13/14 captures and task-choice diagnostics saved. No natural or reserved-test evaluation; document pool remains26,112. Broad goal active.
