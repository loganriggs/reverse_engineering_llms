## H869 — Familiar continuations restore copying/capital competence but not diplomatic binding

Following H868's failed exact-prefix instruction panel, H869 uses the same predeclared country contrasts in repetition copying, diplomatic binding and direct capital continuations. This is an adaptive competence screen, not a fresh confirmation. Copy/capital discovery cells pass; diplomatic binding fails full-vocabulary competence despite perfect binary preference on eligible rows. No intervention claim is made for failed binding cells.

Two repetition formats use country/nation as a repeated key. Binding uses a fixed pair of countries meeting, then one sending a letter or speaking to the other. Capital formats use The capital of COUNTRY is and COUNTRY has its capital in. Pair endpoints change only the source-country token(s); expected answers are the selected country for copying/binding and its capital for retrieval. Same words are used across tasks, but prefixes, counts and positions differ.

|Task|Format|Split|Pairs|Both correct|Full-vocab correct|Pass|
|---|---|---|---:|---:|---:|---|
|copy|A|discovery|8|1.0|0.9375|True|
|copy|A|transfer|1|1.0|1.0|False|
|copy|B|discovery|8|1.0|1.0|True|
|copy|B|transfer|1|1.0|1.0|False|
|binding|A|discovery|8|1.0|0.0|False|
|binding|A|transfer|1|1.0|0.0|False|
|binding|B|discovery|8|1.0|0.0|False|
|binding|B|transfer|1|1.0|0.0|False|
|capital|A|discovery|8|1.0|0.8125|True|
|capital|A|transfer|1|1.0|1.0|False|
|capital|B|discovery|8|1.0|0.8125|True|
|capital|B|transfer|1|1.0|0.0|False|

Only9 lexical pairs survive common eligibility:8 discovery and1 transfer. Four pairs are excluded because capital answers are not single-token. Three additional transfer pairs fail equal-length endpoint tokenization because formatB capital prompts begin directly with a country; sentence-initial tokenization differs from the leading-space country tokens used elsewhere. This is a representation-control problem, not evidence of task failure. No transfer cell has enough pairs to pass the predefined >=4-pair gate.

Next correct that structural alignment by prepending the fixed phrase For reference, to every formatB capital prompt and require exact leading-space country-token IDs across formats. Keep all country pairs, splits, binding/copy prompts and competence thresholds unchanged. This is an explicitly recorded adaptive alignment correction; retain H869 exclusions and outcomes. Binding remains a failed-task control unless it independently satisfies the unchanged gate; do not silently drop the full-vocabulary criterion because binary country preferences are correct.

120 full forwards including replay. Native replay exact; signedAV relativeerror6.57e-7; same-token bus variation1.53e-5. Source-only paired changes audited, sources unchanged, no reserved-test use. Natural-document pool26,112 unchanged. Captures retained for diagnostics; broad goal remains active.
