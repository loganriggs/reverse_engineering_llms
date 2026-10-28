## H909 — Frozen-operator transfer panel establishes competent new-country tests before interventions

The previous turn made progress: H908 found that discovery task-context means process full held-out lexical changes better than small lexical subspaces, while target-margin and distribution fidelity could disagree. H909 establishes the next panel's eligibility and native competence before examining any intervention outcomes. All H908 means at L12/13/14 are hash-frozen and unchanged; no new mean or operator is fitted.

### Panel and eligibility

Use the eight earlier country pairs as regression controls and eight predeclared additional pairs: Netherlands/Austria, Brazil/Mexico, Egypt/Iran, Thailand/Vietnam, Ukraine/Romania, Indonesia/Albania, Serbia/Croatia, Czech Republic/Slovakia. Intended answer continuations are the corresponding conventional language names. These are behavioral probes, not claims that each country is linguistically homogeneous or that every prompt wording captures its legal language status.

Single-token continuation eligibility applies to both task and copying answers in every condition. Indonesia/Albania and Czech Republic/Slovakia fail and are excluded before model evaluation; six new pairs remain. The twelve new eligible countries are absent from the earlier eight language pairs, though several answer languages overlap earlier vocabulary. Novel source countries do not imply all answer directions are novel.

Four task prompts:
- A: “The main language spoken in {country} is” (existing).
- B: “The official language of {country} is” (existing).
- C: “People in {country} mainly speak” (additional).
- D: “The most widely spoken language in {country} is” (additional).

Matched copying uses the earlier word/item repetition formats; A/C and B/D share copy prompts. There are224 evaluated rows but only168 unique prompt strings; copying duplicates must not be treated as independent samples. Old A/B examples are reused regression data. Old C/D change prompts but may include discovery countries. New-country combinations are new to this investigation, not globally untouched or pretraining-disjoint data. No reserved test is touched.

### Predefined competence gate

Each cohort/template/family cell needs at least four eligible pairs, both endpoint binary preferences correct in at least75% of pairs, and exact expected-answer full-vocabulary top1 at least50%. A matched task cell also needs its copy control to pass. No correctness-based endpoint exclusions are applied.

|Cohort|Template|Pairs|Task both-binary|Task exact top1|Copy both-binary|Copy exact top1|Task+copy pass|
|---|---|---:|---:|---:|---:|---:|---|
|old|A|8|1.000|0.938|1.000|1.000|True|
|old|B|8|1.000|1.000|1.000|0.938|True|
|old|C|8|1.000|0.812|1.000|1.000|True|
|old|D|8|1.000|0.938|1.000|0.938|True|
|new|A|6|1.000|0.917|1.000|1.000|True|
|new|B|6|1.000|0.833|0.833|0.917|True|
|new|C|6|1.000|0.417|1.000|1.000|False|
|new|D|6|1.000|0.750|0.833|0.917|True|

All old-country cells pass. New-country A/B/D pass, including a new-country/new-template cell D at75% task top1. New-country C fails with41.7% top1 despite100% binary preference; retain it separately rather than treating binary discrimination as sufficient competence. All copying cells pass, although item-format new-country copying has only83.3% both-endpoint binary correctness.

### Next frozen intervention evaluation

Native logits and normalized final-query MLP inputs at L13/14 are captured for every eligible row, allowing matched donor changes without fitting to these data. Next apply the unchanged pooled-task context map and fixed copying-context controls across the panel, comparing exact midpoint responses, native-linear and quadratic controls. Report new-country A/B/D as competent transfer cells, new-country C as failed-gate diagnostics, and old A/B as reused controls. Keep KL to exact intervention distributions alongside target-margin error. Do not optimize a mean, gate, rank or layer from this panel and then call it fresh confirmation.

This turn produces eligibility/competence evidence and frozen evaluation inputs, not an operator-transfer result. Shared-map canonicality, disjoint-discovery stability, binding pathways, and the broader manifold/featurizer objective remain open.

### Verification

224 native full-model forwards, exactly matching eligible rows. All model/source/frozen-mean hashes unchanged; all margins finite. Exclusion reasons and manifest saved before evaluation, and no intervention effects measured. Captures retain all eligible rows including competence failures. No natural or reserved-test evaluation; document pool remains26,112. Goal active.
