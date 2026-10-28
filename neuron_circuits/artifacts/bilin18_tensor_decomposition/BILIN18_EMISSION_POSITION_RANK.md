## H622 — Final-position direction accounts for most emission effect; rank2 closely preserves distributions but not every answer

The previous turn was progress: H621 established a fixed absolute six-position intervention with cross-country and cross-template effects. H622 asks whether that intervention requires coordinated positions or can be represented more simply. Keep the same24 capital0/capital1/language recipients. Test the full vector, six individual rows, first5/last2/first4 combinations, and unscaled rank1/rank2 SVD approximations, at subtraction doses0.25 and1 and addition dose1. This yields864 intervention evaluations plus24 native baselines, and216 logit/target additivity comparisons. All derived vectors are locked before downstream evaluation; no rank is optimized against logits or answers.

**The last position dominates the causal effect on this panel.** Although it carries51% of squared intervention norm, final-position-only removal gives output-reference KL errors of just0.73%/0.57%/1.13% of omission error for capital0/capital1/language at dose1. The first five rows together retain86.6%/81.7%/90.3% of omission error, and preserve all native capital top answers. This is evidence about a particular final-token intervention, not a claim that the whole model computes at the final position or that other positions are globally unnecessary.

The table reports fidelity to the full intervention at dose1. Relative reference KL is KL(full-intervention distribution || approximation distribution), divided by KL(full-intervention distribution || native distribution), averaged within each family before division. Internal effect error is the mean relative difference in normalized MLP11 input, using the full intervention's shift as denominator. Neither is a causal-effect percentage.

| Panel | Approximation | Internal effect error | Relative reference KL | Native-reference KL | Correct top1 | Intended answer mass |
|---|---|---:|---:|---:|---:|---:|
|capital0|full|0.0000|0.0000|0.8880|0.1250|0.0611|
|capital0|row5|0.1074|0.0073|0.7411|0.5000|0.0758|
|capital0|first5|0.9431|0.8659|0.0041|1.0000|0.3414|
|capital0|last2|0.0783|0.0037|0.7811|0.2500|0.0717|
|capital0|rank1|0.1544|0.0061|0.7604|0.2500|0.0737|
|capital0|rank2|0.0569|0.0013|0.9548|0.1250|0.0553|
|capital1|full|0.0000|0.0000|0.9099|0.0000|0.0228|
|capital1|row5|0.1123|0.0057|0.7797|0.0000|0.0303|
|capital1|first5|0.9287|0.8169|0.0077|1.0000|0.2314|
|capital1|last2|0.0810|0.0027|0.8191|0.0000|0.0279|
|capital1|rank1|0.1442|0.0037|0.8216|0.0000|0.0274|
|capital1|rank2|0.0575|0.0009|0.9540|0.0000|0.0209|
|language|full|0.0000|0.0000|0.1286|0.8750|0.3470|
|language|row5|0.1244|0.0113|0.1104|0.8750|0.3596|
|language|first5|0.9686|0.9031|0.0009|0.8750|0.4785|
|language|last2|0.0986|0.0090|0.1090|0.8750|0.3611|
|language|rank1|0.1983|0.0255|0.0939|0.8750|0.3697|
|language|rank2|0.0651|0.0011|0.1242|0.7500|0.3486|

**Rank2 is a good output-distribution approximation within this panel.** Rank1 retains78.6% of write energy and has dose1 internal errors0.144–0.198; rank2 retains89.6% and has internal errors0.057–0.065. Rank2 relative reference KL is0.0009–0.0013 at full subtraction and0.0006–0.0050 across addition and quarter-dose subtraction. Rank1 is less reliable, especially for language addition (relative reference KL0.089). This illustrates the user's proposed hierarchy: preserving internal vectors, output distributions, and decisions are distinct requirements. The favorable KL does not recover the missing vector geometry or establish a unique canonical coordinate system.

**Approximation can change discrete decisions even at small KL.** On capital0, full removal gives12.5% correct top1 while final-only gives50%, despite relative reference KL0.0073. Rank1 gives25%; rank2 matches12.5%. On language, full removal leaves87.5% correct top1 while rank2 gives75%, despite relative reference KL0.0011. Small differences near a decision boundary matter. None of these approximations should be described as exact functional equivalence, nor as preserving all faithfulness objectives.

**Position effects are mostly additive locally, with finite-dose deviations.** Sum the separate single-row logit shifts and compare with the actual full intervention. Common-offset-invariant logit norms use native-probability weighting. The summed logits have relative cross-term about1.0–1.1% at quarter dose,2.7–3.7% at full subtraction and5.6–6.2% at full addition. Internal cross-terms are about1.1–1.2%,4.5–4.9% and4.3–5.0%, respectively. Exact KL between the full distribution and additive-logit prediction is0.00022–0.00080 nats at full subtraction. These are direct additivity tests; comparing KL sums alone would not measure nonlinearity.

| Panel | Signed removal dose | Relative logit cross-term | Relative internal cross-term | Additive-logit prediction KL |
|---|---:|---:|---:|---:|
|capital0|-1.0|0.056354|0.043146|0.000225|
|capital0|0.25|0.011254|0.011444|0.000003|
|capital0|1.0|0.037469|0.044932|0.000801|
|capital1|-1.0|0.057860|0.043425|0.000571|
|capital1|0.25|0.009616|0.011356|0.000004|
|capital1|1.0|0.027057|0.049041|0.000752|
|language|-1.0|0.062435|0.050455|0.000063|
|language|0.25|0.010975|0.012473|0.000000|
|language|1.0|0.037263|0.046126|0.000215|

The sign convention is positive=subtraction, negative=addition. First5+last and first4+last2 comparisons are also stored. All vectors are unscaled truncations or approximations; no norm restoration hides lost components. The main capital0 summary combines four fit and four excluded countries; item-level split labels remain available. These prompts are familiar from prior experiments, so this tests sufficiency within an adaptive panel rather than independent generalization.

The strongest current candidate unit is now one fixed1152-dimensional direction at Attention5's final output position, with a finite approximation error. Rank2 across positions provides a more faithful distributional representation. These are competing useful descriptions at different tolerances, not proof that the model internally stores either as an indivisible semantic feature. All tested recipients end in "is" and have closely matched positions. The final direction could act through a final-token-specific normalization or readout mechanism rather than a general answer variable.

Next use the final-position-only vector to escape the six-token stencil constraint. Freeze it and test varied sequence lengths, different final tokens and relation scaffolds, with fixed signed doses and random equal-norm controls. Predeclare prompts and answer sets, preserve competence failures, and score short answer continuations including quote-prefixed forms so next-token format alone does not dominate interpretation. Distinguish generic answer emission from relation identity and quantify effects on the full candidate distribution. No new direction fit should be used to rescue transfer failures.

Audits: all source/manifest/derived-vector/runner hashes unchanged; single rows sum exactly to full vector; full-intervention output accuracy, KL and intended-token mass reproduce H621 within1e-5;864 evaluations,216 factorial comparisons finite. Native baselines computed for24 recipients; no new natural-text documents, FineWeb remains26,112. Broad research goal active and incomplete.
