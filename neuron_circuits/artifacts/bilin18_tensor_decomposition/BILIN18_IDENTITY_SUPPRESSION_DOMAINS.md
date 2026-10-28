## H670 — Input-linked suppression generalizes, including to associated answers

H669 found current-token suppression in month/number contexts. H670 extends the same finite probe to nouns/plurals, adjectives/antonyms, countries/capitals and months/succession. It uses128 existing held-out single-token-input prompts: four items per family, four forms, copying and task instructions. No items or directions are fitted or selected from the new results. This is a retrospective domain comparison on an existing panel, not an unseen-item replication.

At layer14 head4's final attention write, add alpha times the receiver's projected effective input value, using fixed alphas−0.1,−0.05,−0.025,0,+0.025,+0.05,+0.1. The rest of the model runs normally. Reuse seed669's unit random direction, scaled to each receiver value norm, at±0.05. The actual input-token value is not assumed context invariant.

Readouts distinguish the raw input token from the requested task answer even in copying prompts. Also measure the average logit of the seven other input identities in the same eight-item family, excluding the current item. Those margins guard against interpreting a uniform logit shift as input-selective suppression. The alternative set is limited to this panel, not the whole vocabulary.

**Central finite logit slopes at alpha0.05:** slope=(logit(+alpha)−logit(−alpha))/(2alpha), per unit coefficient.

| Family | Context | Input token | Task answer | Other inputs mean | Answer−input | Other inputs−input |
|---|---|---:|---:|---:|---:|---:|
|plural|copy|-2.7319|-2.6629|0.3612|0.0690|3.0931|
|plural|task|-2.6847|-2.5467|0.3748|0.1381|3.0595|
|antonym|copy|-1.7427|-0.0989|-0.0319|1.6438|1.7108|
|antonym|task|-1.7270|-0.0753|-0.0315|1.6517|1.6955|
|capital|copy|-3.2021|-1.8290|0.6351|1.3730|3.8371|
|capital|task|-3.2057|-1.7402|0.7076|1.4656|3.9133|
|successor|copy|-2.4441|0.6216|0.3250|3.0658|2.7692|
|successor|task|-2.4379|0.6282|0.3699|3.0662|2.8078|

Every one of128 tested contexts has a negative input-token slope and positive other-input-minus-current margin slope. Thus input-linked suppression extends beyond ordered sequences and is not simply a common logit offset. The pattern appears under both copying and task instructions. Its magnitude is not normalized across domains because native value norms differ.

The requested answer is not generally boosted in absolute logit terms. Plural answers fall almost as much as singular inputs (about−2.55 to−2.66 versus−2.68 to−2.73); only21/32 noun contexts have a positive answer-minus-input slope. Capitals fall substantially (about−1.74 to−1.83) along with their country inputs (about−3.20). Antonym answer slopes are slightly negative. Month-successor answers instead rise. Therefore the unit is not well described as simply removing the incorrect input while selecting the correct task answer.

A broader interpretation is suppression of content associated with the input value, with uneven effects on related alternatives. Singular/plural and country/capital relations suggest that the affected content can extend beyond the exact token identity. This remains a hypothesis about one tested direction, not proof of a general semantic cluster or feature ontology.

**Norm-matched random control margins, same strength:**

| Family | Context | Answer−input | Other inputs−input |
|---|---|---:|---:|
|plural|copy|-0.1930|-0.0304|
|plural|task|-0.2055|-0.0231|
|antonym|copy|-0.1275|-0.0396|
|antonym|task|-0.1515|-0.0714|
|capital|copy|0.1470|-0.0564|
|capital|task|0.0819|-0.0715|
|successor|copy|-0.0520|0.1835|
|successor|task|-0.0033|0.2100|

These mean margins are much smaller and have mixed signs. One random direction is only a limited control, not a null distribution or a statistical significance claim. No superiority of all value directions over all random directions is inferred.

The next question is where the suppression originates. Compare the value vector's direct contribution through the residual skip path and final normalized/saturated readout with the measured effect after remaining MLP/attention processing. A strong direct alignment would suggest the projected value already encodes suppressive output structure; a large downstream change would implicate later transformations. Use full-vocabulary and selected-token comparisons, explicitly include residual scaling, and keep that linear path diagnostic distinct from the actual finite intervention. This is more informative than naming a semantic unit from the sign pattern alone.

The original model, native receiver values and context-dependent gate are still required. These probes do not establish head necessity, a unique canonical basis, a full task algorithm or improved task correctness. Previous fidelity improvements are compatible with adjusting the strength of broad content suppression.

Audits:1152 whole-model forwards (128 prompts×9 fixed conditions). Native replay maximum absolute logit error1.91e-5. All source/factor/runner hashes unchanged; finite effects. Single input-token selection asserted; task answer distinguished from copying target; exactly seven non-current input identities per family verified. All strengths, items and contexts retained. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
