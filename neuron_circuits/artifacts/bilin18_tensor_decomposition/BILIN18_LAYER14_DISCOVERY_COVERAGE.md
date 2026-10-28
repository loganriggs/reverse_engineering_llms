## H662 — Broader discovery coverage only modestly reduces layer14 error

H661 localized much of the unresolved month-succession mean-prediction error to layer14 attention. H662 checks whether the ordered first-four-item discovery set explains that result. Keep every other layer's second-wording mean fixed and retain the original96 evaluated cases (items4–7 in each family/form). Change only the fitted layer14 copy-to-task attention shift:

- Ordered4: original item indices0,1,2,3.
- Alternating4: evaluated index plus offsets1,3,5,7 modulo8. This is balanced over alternating item indices, not a guarantee of semantic balance.
- Leave-one-out7: all seven items except the evaluated item.
- Exact14: donor-informed layer14 output replacement, the diagnostic reference from H661.

All fits exclude the evaluated item and use the same form. No other forms of that evaluated item enter the fit. Alternative fits reuse other previously evaluated items, so they are retrospective cross-validation, not an independent blind replication. The same prompts, task labels, output targets, and boundary/other-attention predictions remain fixed. Both predicted and exact boundary states are evaluated.

**KL(task||edited), predicted boundary:**

| Family | Ordered4 | Alternating4 | Leave-one-out7 | Exact14 |
|---|---:|---:|---:|---:|
|addition|0.007949|0.008273|0.008291|0.004864|
|agreement|0.005285|0.005242|0.005234|0.005048|
|antonym|0.011206|0.011370|0.011265|0.008854|
|capital|0.017518|0.018000|0.017982|0.008596|
|plural|0.010843|0.010311|0.010535|0.008744|
|successor|0.086528|0.080527|0.081595|0.021949|

**Exact boundary, to isolate attention prediction:**

| Family | Ordered4 | Alternating4 | Leave-one-out7 | Exact14 |
|---|---:|---:|---:|---:|
|addition|0.001671|0.001842|0.001846|0.000315|
|agreement|0.001706|0.001715|0.001660|0.001382|
|antonym|0.009779|0.010156|0.010010|0.005359|
|capital|0.013138|0.013502|0.013569|0.003899|
|plural|0.007369|0.006993|0.007050|0.003546|
|successor|0.081667|0.077749|0.077863|0.014042|

Month-succession KL improves only modestly:0.08653→0.08053 with alternating4 and0.08160 with leave-one-out7, compared with0.02195 for exact14. Under exact boundary,0.08167 becomes about0.0778 while exact14 reaches0.01404. The original ordered split is therefore not the sole source of the residual. These schemes change both coverage and composition; seven-item averaging is not guaranteed to dominate four-item averaging.

Other tasks are mixed. Plurals improve slightly; addition and capitals worsen slightly, while agreement changes little. More discovery examples do not automatically produce a better functional mean. None of this establishes irreducible item complexity or rules out larger, more representative discovery sets and conditional prediction.

**Layer14 geometric relative RMSE**, sqrt(sum squared shift-prediction error / sum squared true shift norm), reported separately from output KL:

| Family | Ordered4 | Alternating4 | Leave-one-out7 |
|---|---:|---:|---:|
|addition|0.1555|0.1655|0.1648|
|agreement|0.1711|0.1868|0.1618|
|antonym|0.2642|0.3011|0.2634|
|capital|0.3522|0.3536|0.3440|
|plural|0.2056|0.1891|0.1841|
|successor|0.7855|0.7102|0.6949|

Succession's geometric error improves0.7855→0.6949 with seven-item fitting, yet most of the functional gap to exact14 remains. Geometry and output fidelity can respond differently. The representation may require an item-conditioned component, or the remaining geometric errors may align with particularly sensitive output directions. This experiment does not choose between those explanations.

The next mechanistic test is within layer14: decompose its attention output into projected head writes and perform single-head exact residual repairs plus leave-one-predicted controls. Fit head-level discovery means so the head decomposition sums back to the current mean prediction, and verify reconstruction before interpreting repairs. Keep the other layers and boundary settings fixed, and retain all six families rather than selecting only the conspicuous month task. This can distinguish a concentrated head-level failure from distributed residual variation before adding a more elaborate predictor. A subsequent query/key versus value decomposition can then address how any identified head depends on the item.

These results concern approximation of a possibly incorrect task-prompt output distribution, not demonstrated execution of month succession. Exact next-token and candidate-score limitations from the held-out panel remain. The original network, known family/form, and native receiver baseline are still required. No canonical feature basis or minimal semantic unit is established.

Audits:768 completed whole-model forwards;1152 cached attention-stage evaluations in the successful run. Two preflights each completed1152 cached attention evaluations and stopped before any edited whole-model forwards: changing batch composition shifted a reconstructed mean by0.0003204 and violated the1e-5 replay check. The successful implementation separates discovery and transfer batching, reproducing the original discovery mean exactly without relaxing tolerance. All four ordered/exact14 baseline logit arrays match H661 exactly. Evaluated-item exclusion is asserted for every fit; source and runner hashes unchanged; KL values finite. Total cached evaluations including stopped preflights:3456. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
