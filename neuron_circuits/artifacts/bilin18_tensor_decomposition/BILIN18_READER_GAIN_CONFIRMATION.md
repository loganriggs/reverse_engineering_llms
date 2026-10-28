## H874 — Frozen downstream gains transfer across lexical combinations but fail across a competent new format

The prior goal turn made progress by freezing H872's six context gains and defining a no-refit check. H873 then screened new countrypair combinations and continuation variants. H874 tests the frozen gains. ForL13H0/L14H4, gains transfer well to new pairs in familiarA/B formats, but become harmful in the competent newC format. This contradicts a format-independent copying-to-capital scalar-gain hypothesis. It does not invalidate the exact source-value/routing factorization at the reader.

### Frozen confirmation protocol

All12 new pair combinations of24 familiar countries, fourformats, twofamilies and192endpoints retained. H872 discoverygains and reader choices were frozen and hashed before new outcomes. Identity and predictednative source-midpoint head edits are independently audited against direct mixed-value edits. Positive/negative common-amplitude writes use the same country direction and amplitude in matched copy/capital contexts.

As inH872, orient responses to countryA→B and average both endpoints within each lexicalpair. Familiar-format results averageA/B; newC and diagnosticD are separate. The measured copying-context symmetric finite-amplitude response is the predictor input; multiply it by the frozen gain to predict the same country or capital output contrast's response in the capital context. This is conditional response transfer, not intervention-free prediction of native model behavior. Native routing is measured anew, not predicted by the gain model.

### Relative response RMSE, no refitting

|Reader|Output contrast|New pairs, A/B: no gain|New pairs, A/B: frozen gain|New pairs, C: no gain|New pairs, C: frozen gain|
|---|---|---:|---:|---:|---:|
|L13H0|country|0.212|0.076|0.143|0.276|
|L13H0|capital|0.137|0.082|0.221|0.298|
|L14H4|country|0.219|0.075|0.152|0.271|
|L14H4|capital|0.309|0.094|0.108|0.272|
|L8H3|country|0.325|0.292|0.280|0.289|
|L8H3|capital|0.258|0.275|0.361|0.395|

Late-reader frozen gains achieve7.5–9.4% error on new pairs in oldformats, substantially below no-gain errors13.7–30.9%. In new competentC, their errors rise to27.1–29.8%, worse than no-gain errors10.8–22.1%. Allfour late-reader mappings show this failure, with saved pair-bootstrap intervals. For exampleL13H0 country-response RMSE rises from.076[.054,.098] inA/B to.276[.244,.303] inC. L8H3 remains less accurately captured overall, with27.5–39.5% frozen-gain errors across these cohorts.

CapitalD failed the predefined full-vocabulary competence gate inH873 and is retained only as a diagnostic. Its late-reader frozen-gain errors are19.6–22.4%, also worse than no gain, but this is not positive task-level generalization evidence. No individual failing cases were discarded and no replacement gains were fitted.

High response correlation does not rescue the failed amplitude prediction. For exampleL13H0 country pair correlation remainsabout.957 inC while frozen-gain RMSE is.276. A relationship can preserve lexical ordering yet have the wrong scale. Familiar-format averaging may also smooth variation; per-format mechanisms need investigation rather than treating the pooledA/B gain as a universal constant.

### Hypothesis refresh

|#|Hypothesis|Current evidence / next discrimination|
|---:|---|---|
|1|The source bus is a fixed token-local content map|Supported by static-value and same-token audits.|
|2|A reader's source-value edit factors into content × signed routing|Exact within floating-point tolerance for tested interventions.|
|3|Task effects can reverse through routing sign|Causally supported by H872 sign controls.|
|4|One late-reader context gain is format independent|Contradicted by competent newC.|
|5|Frozen gains generalize to new lexical pairings in familiar formats|Supported for late readers on these12combinations.|
|6|Final residual RMS explains much of format gain variation|Plausible, not tested; measure native norms next.|
|7|RMS tangential projection matters beyond a scalar norm ratio|Plausible; compare full normalization formula with norm-only model.|
|8|Final logit-cap saturation changes readout gain|Plausible; include the actual30tanh readout in analytic controls.|
|9|Later MLP responses cause the remaining format dependence|Plausible; isolate frozen-update versus live downstream effects.|
|10|Later attention/routing responses cause it|Plausible; separate from MLP changes after basic normalization test.|
|11|Finite edit amplitude rather than local sensitivity causes failures|Possible; use an amplitude sweep only if analytic controls leave ambiguity.|
|12|Head direction alone determines downstream behavior|Weakened by equal-write context dependence.|
|13|Earlier readers have more context-dependent responses because more layers remain|Consistent but confounded with head identity.|
|14|Source occurrence count explains cross-task effects|Affects routing magnitude, but not the whole fixed-write downstream gap.|
|15|Country versus capital output contrasts need different gains|Observed, so output metric must remain explicit.|
|16|Binary competence suffices for task interpretation|Rejected by failedbinding/capitalD full-vocabulary checks.|
|17|Many featurizer atoms imply a high-dimensional task computation|Weakened by within-union rank1 intervention results.|
|18|A scalar/vector feature can be globally labeled by one task|Weakened by shared readers and context-dependent sign.|
|19|An explicit shared operation can transfer more reliably than a reconstruction group|Promising, but unsupervised discovery and broad transfer remain unverified.|
|20|A data-based canonical description requires a context-dependent downstream metric|Strengthened; uniqueness/minimality and practical learned recovery remain open.|

### Next discriminating experiment

Keep the failed frozen-gain predictions unchanged. Measure native final residual states on the old and new contexts. First test the parameter-free RMS ratio as a candidate context gain. Then predict the finite head-write response through the residual skip path with later updates held fixed, using the actual final RMS normalization, unembedding andlogitcap. This analytic baseline separates output normalization/readout effects from changes induced in later computations. It is a counterfactual frozen-update model, not the actual model unless validated against live interventions.

If normalization accounts for the format shift, a conditional response metric may be a better representation target than fitted task gains. If not, localize later computation effects before adding model capacity. Do not silently fit format-specific gains and call that generalization. The overall goal still includes task-blind featurizer recovery, manifold verification and broader tasks; these controlled country experiments establish only a scoped mechanism.

### Verification and limits

2,880 whole-model forwards. Exact native replay; predicted-head versus direct mixed-value intervention logits agree within1.14e-05. Matched context source contrasts differ by at most1.53e-05; within-context repeated-source contrasts are exact. Routing replay error2.38e-07; one readerwrapper call perforward, all metrics finite, hashes including frozen gains unchanged. No reserved-test use;26,112-document pool unchanged. Pair-bootstrap4096 uncorrected descriptive intervals; new pairs share familiar tokens and belong to a small controlled distribution. No claim of globally untouched research data or general language coverage. Broad goal remains active.
