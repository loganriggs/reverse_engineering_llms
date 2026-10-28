## H870 — A matched-country comparison is competent for copying and capital retrieval; binding remains a failed control

This turn made concrete progress through three recorded screens. H868's exact-common-prefix instruction panel failed competence. H869's familiar continuations restored copy/capital performance but revealed sentence-initial tokenization mismatches. H870 corrects that structural mismatch with a fixed For reference, prefix before formatB capital prompts and requires the exact same leading-space country tokens across tasks. The country list, split, copy/binding prompts and competence thresholds remain unchanged.

Twelve lexical pairs survive the predeclared common single-token answer eligibility:8 discovery and4 transfer,144 contexts. Four pairs remain excluded because their capital answers require multiple tokens. Every eligible pair has the same two country-token IDs across all task/format contexts. Copying changes two repeated source occurrences, capital retrieval one, and binding the single giver country; the binding input-contrast sign reverses relative to answer identity. Source positions/counts and prefixes are not identical across task families, so routing remains an explicit variable.

### Native competence

The fixed gate requires >=4 eligible pairs, >=75% of pairs correct at both binary endpoints, and >=50% intended-token full-vocabulary top1. All individual examples remain included in each cell; there is no correctness-based filtering.

|Task|Format|Split|Pairs|Both endpoints correct|Full-vocab correct|Pass|
|---|---|---|---:|---:|---:|---|
|copy|A|discovery|8|1.0|0.9375|True|
|copy|A|transfer|4|0.75|0.875|True|
|copy|B|discovery|8|1.0|1.0|True|
|copy|B|transfer|4|1.0|0.875|True|
|binding|A|discovery|8|1.0|0.0|False|
|binding|A|transfer|4|1.0|0.0|False|
|binding|B|discovery|8|1.0|0.0|False|
|binding|B|transfer|4|1.0|0.0|False|
|capital|A|discovery|8|1.0|0.8125|True|
|capital|A|transfer|4|1.0|0.75|True|
|capital|B|discovery|8|1.0|0.9375|True|
|capital|B|transfer|4|1.0|0.875|True|

Copying and capital retrieval pass all8 task/format/split cells. Diplomatic binding fails all4: binary country preference is100%, yet full-vocabulary next-token accuracy is0%. This means binary forced-choice competence would be misleading here. It does not establish that the model cannot perform binding generally or could not name the country later in a multi-token continuation; the previously competent human-name panel remains distinct evidence.

### Research direction

We now have a controlled lexical comparison for copying versus associated capital retrieval, with familiar continuation formats and exact shared source-token identities. It is not a three-task success. Keep the failed country-binding cells as negative controls and the prior competent human-name binding tasks as separate checks, without pretending their vocabularies match.

Freeze this panel and the previously fixed readersL13H0,L14H4,L8H3. Next test source-value midpoint erasure at those readers, separating final-query readout from all-query effects. Use captured native signed routing to predict the exact value-write perturbation from the static country-bus difference, with no fitted scale or sign. Compare effects on country identity and capital association across task contexts, keeping all eligible endpoints. This tests whether a common stored direction is used by different contextual readouts; successful source-value algebra alone is not evidence of a semantic manifold or shared full computation.

H739 already established static bus maps and some cross-context substitutions. The added control here is matched country contrasts across explicitly competent repetition and capital tasks, with the new frozen reader set and failed binding condition disclosed. Reconcile any result with that prior work. No new architecture training is justified merely by this competence screen.

### Verification and freshness

156 whole-model forwards including exact native replay. SignedAV reconstruction relativeerror6.57e-7; same-token source-bus variation1.53e-5 raw units. An independent audit confirms each lexicalpair uses exactly the same two country-token IDs across allfamilies/formats; within-task endpoints differ only at declared country positions. Inputs unchanged, outputs finite, no reserved-test use. Natural-document pool26,112 unchanged.

The three screens were adaptive. Transfer countries were withheld by the declared split but have already appeared in the screening sequence and may have prior research exposure. They are not untouched global research holdouts. Subsequent reader interventions can be diagnostic on this frozen panel, but stronger generalization claims require new combinations or data. Broad goal remains active.
