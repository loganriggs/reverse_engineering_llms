## H873 — New lexical combinations and fixed continuation variants: competence screen

H872 completed coefficient interchange and froze six discovery-fit context gains. H873 prospectively screens a new combination panel before assessing gain predictions. All24 previously eligible country/capital items are paired deterministically across the two halves of their ordered list, producing12 countrypair contrasts absent fromH870. Individual tokens remain familiar. No pair repeats, no exclusions, and all192 endpoints are retained.

FormatsA/B reuse H870 continuations. NewC/D copy keys are place/location in repeated sequences. CapitalC is The capital city of COUNTRY is; capitalD is In COUNTRY, the capital is. Copy/capital use exactly the same countrytoken contrasts but different sourcepositions/counts. Binding's existing failed-country and competent-human-name panels remain separate controls; they are not relabeled as successful matched-country evidence.

The gate remains >=4 eligible pairs,>=75% both-endpoint binary correctness,>=50% intended-token full-vocabulary top1. H872 gains and the3reader targets were frozen and hashed before outcomes.

|Task|Format|Pairs|Both correct|Full-vocab correct|Pass|
|---|---|---:|---:|---:|---|
|copy|A|12|1.0|0.9166666666666666|True|
|copy|B|12|1.0|0.9583333333333334|True|
|copy|C|12|1.0|1.0|True|
|copy|D|12|1.0|1.0|True|
|capital|A|12|1.0|0.7916666666666666|True|
|capital|B|12|1.0|0.9166666666666666|True|
|capital|C|12|1.0|0.875|True|
|capital|D|12|1.0|0.4583333333333333|False|

Seven of eight cells pass. Both tasks are competent inA/B andnewC. CapitalD has100% binary preference but45.8% full-vocabulary top1 and fails the unchanged gate. Do not use its mechanistic result as positive evidence on a competent capital task; retain it as a failed-format diagnostic. Primary gain-confirmation comparisons should distinguish old-format/new-pairA/B, new-format/new-pairC, and diagnosticD without per-example correctness filtering.

204 full forwards including exact native replay. SignedAV reconstruction relativeerror1.34e-6; same-token bus variation1.53e-5. Paired inputs differ only atdeclared sourcecountrypositions, input hashes unchanged, metrics finite. No reserved-test use and26,112-document naturalpool unchanged. Freshness is new combinations andC/D templates relative toH870/H872, not newtokens, pretraining-disjoint material or guaranteed globally unseen research prompts.

Next evaluate the existing six context gains with no refitting using matched positive/negative headwrites atL13H0,L14H4,L8H3. The predictor will use the measured copying-context response to predict the capital-context response; it is a conditional response-transfer test, not an intervention-free predictor of native model behavior. Broad goal remains active.
