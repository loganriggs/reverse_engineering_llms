## H871 — Matched country directions reveal shared reader use and context-dependent sign reversal

H868–870 established a matched-country panel competent for copying and capital retrieval, while country-as-agent binding failed full-vocabulary competence. H871 now performs frozen-reader source-value interventions on every eligible endpoint. L13H0 contributes strongly to both copying and capital retrieval. L14H4's contribution supports capital retrieval but opposes the copying margin, accompanied by a context-dependent change in effective signed routing. This is stronger evidence for reusable content maps with contextual coefficients than for a fixed task-specific head label.

### Exact intervention and algebra

Retain all12 country pairs, twoformats, threefamilies and144 endpoints at frozenL13H0,L14H4,L8H3. Native local values and query/key computations are unchanged at the selected reader. Add half the donor-minus-recipient static bus difference, multiplied by that reader's value-mixing coefficient lambda, only at the changed source positions in its mixed-value tensor. The persistent bus itself remains native, so other readers do not receive edited bus inputs.

Apply the edit at either every query position or only the final query. Independently predict the final-query head perturbation by

    delta_head = gamma(context) × [B(donor token) − B(recipient token)]
    gamma(context) = lambda / 2 × sum(native signed routing at changed source positions).

B is the fixed layer0 token-value map for that head index. Copying has two changed source occurrences, capital one. Within each context their bus contrasts are exactly identical, so summing routing coefficients is exact. The effective coefficient gamma is signed and is not a probability. Its sign includes lambda as well as the signed attention pattern. No signs, scales or directions are fitted to task outcomes.

### Results

Values are native paired answer-margin fractions removed. Negative values mean that midpoint-erasing the source-value contrast increases the intended behavioral distinction. These are not accuracy losses or unique mediation fractions. Diplomatic binding rows are retained only as failed-task diagnostics.

|Task|Reader|Final-query gap removed|All-query gap removed|Mean signed gamma|
|---|---|---:|---:|---:|
|copy|L13H0|54.1%|55.6%|-0.612|
|copy|L14H4|-46.5%|-47.4%|0.389|
|copy|L8H3|26.7%|26.8%|-0.419|
|capital|L13H0|49.8%|50.1%|-1.282|
|capital|L14H4|8.7%|8.7%|-0.256|
|capital|L8H3|1.5%|1.6%|-0.075|
|binding|L13H0|13.7%|13.5%|0.328|
|binding|L14H4|57.5%|57.7%|1.035|
|binding|L8H3|1.6%|1.6%|0.058|

For competent copying, L13H0 removes54.1% of the pair gap at the final query, compared with49.8% for capital retrieval. Its effects remain positive on transfer rows,55.1% and47.5%, respectively. The earlier apparent capital specialization is therefore not a general property of that reader independent of vocabulary and prompt format. This experiment does not separately identify how much of the change from the earlier panel is lexical versus format-driven.

L14H4 removes8.7% of the capital gap, but its erasure increases copying's gap by46.5%. The pointwise lexical-cluster intervals are[4.7%,13.3%] and[-60.4%,-36.8%]. The negative copying effect appears in both discovery and transfer subsets. All48 copying endpoint coefficients are positive, whereas39/48 capital coefficients are negative. For a fixed country pair at this reader, the source-value contrast is the same; contextual coefficients can reverse the direction of the head perturbation. This is consistent with the observed effect reversal, but downstream state can still modulate the effect of a head vector.

L8H3 is more copy-weighted on this panel:26.7% versus1.5% capital gap removal. Final-query and all-query edits give similar aggregate effects for these frozen targets, suggesting that much of their measured source-value influence is expressed at the final query here. This is not a general claim that nonfinal reads are irrelevant.

The country and associated-capital output contrasts are both recorded in every context. AtL13H0, final-query erasure reduces country/capital paired gaps by3.13/3.09 raw logit units in copying contexts and5.83/5.80 in capital contexts. These coupled changes support a shared lexical/association readout hypothesis, but do not identify where the association was learned or prove identical downstream computation. Cross-output raw gaps are not normalized when their native denominators could be small.

### What this adds, and what remains open

H739 already demonstrated a static bus map and cross-context substitutions. H871 adds frozen-reader tests on matched country-token contrasts in independently screened competent copying and capital continuations, including theL14H4 sign reversal and a final-query/all-query comparison. It does not discover the signed-attention architecture anew.

The current best description is a fixed content direction multiplied by a context-dependent signed coefficient, followed by a reader-specific write map and downstream computation. That is a computational relation, not merely a cluster of activation vectors. It motivates testing explicit shared computations in addition to dictionary geometry. Signed scalar/block encoders can already express negative coordinates; this result does not show that they are incapable of representing the states. The missing test is whether a learned representation exposes the reusable relation reliably without task-supervised selection.

Next freeze same-pair source directions and interchange only effective routing coefficients between copying and capital contexts. Compare native coefficient, matched foreign-context coefficient, sign-flip and magnitude-matched controls at the same reader and same recipient state. This will test how much behavioral variation follows the coefficient rather than downstream context. Interpret such interventions as counterfactual head-write operations, not necessarily natural attainable attention patterns. Retain both formats/splits, failed-binding controls and prior native task competence, without selecting favorable pairs. New combinations are still needed for stronger generalization.

### Verification and scope

1,728 full-model forwards, exact native logit replay. Source contrast equality across repeated positions is exact. Signed-routing/static-value head prediction error2.45e-07 relative to nativehead norm; predicted versus direct final-query intervention logits differ by at most1.05e-05. Routing replay error2.38e-07; eachreader wrapper invoked exactly once perforward. Sources unchanged, metrics finite, no reserved-test use;26,112-document pool unchanged. Clusterbootstrap4096 resamples keeps bothformats together, with12 total/8discovery/4transfer lexicalclusters; intervals uncorrected on an adaptive reusedpanel. Binding remains noncompetent under the full-vocabulary criterion. Broad goal remains active.
