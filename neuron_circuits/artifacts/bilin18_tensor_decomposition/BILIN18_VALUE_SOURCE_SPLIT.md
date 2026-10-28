## H672 — Carried value supplies most of the suppressive response

H671 located suppressive alignment in the projected input value before later branches. H672 separates that value into its actual weighted local and persistent-bus contributions at layer14 head4. The same128 previously inspected contexts cover nouns/plurals, adjectives/antonyms, countries/capitals and months/succession, with copy/task instructions, four items and four forms. No fitting or item selection was performed.

The learned mixing scalar is4.625. Effective value equals−3.625 times the local value plus4.625 times the persistent bus value. This is an unconstrained signed combination, not a convex mixture or a fraction of information. Results below retain these native weights; they do not compare the unweighted sources.

Each weighted projected component was added at the final attention write at strengths±0.025 and±0.05, followed by normal downstream computation. A zero-edit replay was included. Direct residual/readout derivatives were computed separately as in H671.

| Family | Component | Full input slope | Full answer slope | Direct input slope | Direct answer slope |
|---|---|---:|---:|---:|---:|
|plural|local|-0.1194|-0.0961|-0.0408|-0.0328|
|plural|bus|-2.5889|-2.5088|-4.0980|-3.9205|
|antonym|local|-0.1441|-0.0611|-0.0726|0.0036|
|antonym|bus|-1.5908|-0.0261|-2.0555|-0.3224|
|capital|local|-0.0848|-0.1005|0.0745|0.0201|
|capital|bus|-3.1191|-1.6842|-3.9786|-2.3210|
|successor|local|-0.0876|-0.0561|0.0030|0.0203|
|successor|bus|-2.3534|0.6810|-2.5881|0.2060|

The carried component suppresses the input in all128 contexts. The local component has a smaller suppressive full response in127/128 contexts. On family-mean input slopes, the carried contribution accounts for roughly92–97% of the summed signed response; this arithmetic comparison is not a causal share or necessity estimate. Both contributions reinforce suppression on average. Plural/capital answer suppression also comes primarily from the carried contribution. For months, the carried component boosts the next-month answer while the local component modestly suppresses it.

This shifts the leading hypothesis toward a late context-dependent routing gate acting on content carried from an earlier value representation. A locally constructed layer14 semantic operator is less consistent with these observations. However, the layer14 output projection still determines how the carried vector maps into residual space: the experiment does not prove that the earlier representation alone encodes suppression. The next discriminating test should compare how different heads' output projections read the same carried representation, including sign, specificity and intervention magnitude. That separates shared carried content from head-specific suppressive readout geometry.

Audits:1152 whole-model forwards;128 contexts×9 conditions. Native replay maximum absolute logit error1.907e-5. Weighted components reconstruct the saved effective projected vector with maximum relative error1.319e-7. The sum of full component slopes reconstructs the previously measured combined slope with maximum native-probability-weighted centered relative error1.972e-4. Step-size discrepancy maxima are0.001796 local and0.000255 bus, supporting the local additive interpretation. All tracked source hashes unchanged and recorded effects finite. The prior H670 finite-response capture is used as an additional comparison source but was not included in this runner's before/after hash list; it is retained as the authoritative saved reference. No new FineWeb documents;26,112 unique documents unchanged.

Limits: repeatedly inspected panel, weighted derivative decomposition rather than finite removal or necessity, four single-token domains and one head. Output behavior and task correctness remain distinct. No unique canonical unit or full algorithm has been established. Broad research goal remains active and incomplete.
