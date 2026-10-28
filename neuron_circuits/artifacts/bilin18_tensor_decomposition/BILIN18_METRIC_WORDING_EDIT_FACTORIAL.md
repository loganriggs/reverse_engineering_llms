## H603 — Wording and context-edit changes each break frozen-span transfer

H602 changed wording and context edits together. This matched factorial separates them on H601's held-out item IDs 4–7 across all five tasks. Cross original/new wording with original nuisance edit/For-the-record prefix. Preserve the lexical content of the original edit across wordings: letter→message for copying, folder→notebook for binding, window→door for agreement, Fact-prefix for recall, performance→show for sentiment. New wordings change sentence construction while retaining those edit pairs. The prefix condition prepends For the record and lowercases the original initial character; it is not a pure token-position intervention.

Freeze H601's broad64 basis and number/multi-signal metrics. Use number/task/random signals at equal Attention5 norm with both signs. Template codes: 0=old wording/old edit, 1=old wording/prefix, 2=new wording/old edit, 3=new wording/prefix. No fitting or outcome filtering. The item split excludes fitting items but the old cells were already observed; this is an adaptive diagnostic, not an independent confirmation.

The manifest contains 320 prompt records and 222 unique exact strings. Recipient prompts repeat across edit conditions. There are 960 capture forwards (480 recipient/signal/sign records before repetition across edit conditions), followed by 5,760 interventions.


Copying and agreement remain fully competent in all four conditions. Binding also passes recipient competence in both wordings; its prefixed donors do not. Sentiment passes at 0.625 for old wording and 0.500 for new wording; some prefixed donors are weak. Recall's new wording fails full-vocabulary competence despite candidate correctness. Gate labels therefore apply to recipients, not proof that both ends of every context switch perform the intended task.

**Both changes independently reduce fidelity.** With familiar wording held fixed, replacing the original edit by the prefix increases relative KL in every task/signal cell. With the original edit held fixed, changing wording also increases relative KL in every cell. These are paired panel findings, not independent statistical replications.

| Task | Signal | Old wording / old edit KL | Old wording / prefix KL | New wording / old edit KL | New wording / prefix KL |
|---|---|---:|---:|---:|---:|
|sequence_copy|number|0.0608|0.6802|0.5798|0.7136|
|sequence_copy|task|0.0758|0.6389|0.3812|0.7187|
|sequence_copy|random|0.0741|0.7147|0.5017|0.6728|
|agreement|number|0.0677|0.3864|0.5100|0.3211|
|agreement|task|0.0727|0.3981|0.4731|0.3657|
|agreement|random|0.0681|0.7691|0.5920|0.6816|
|capital_recall|number|0.0473|0.4357|0.2791|0.4438|
|capital_recall|task|0.0388|0.3522|0.3223|0.6650|
|capital_recall|random|0.0513|0.4236|0.3065|0.5409|
|entity_binding|number|0.0338|0.7946|0.3481|0.9068|
|entity_binding|task|0.0317|0.8493|0.2784|0.8066|
|entity_binding|random|0.0378|0.8381|0.2265|0.9529|
|sentiment|number|0.0321|0.7322|0.0957|0.6169|
|sentiment|task|0.0185|0.8709|0.0737|0.7585|
|sentiment|random|0.0277|0.8885|0.0976|0.9026|

These are number-metric KL errors relative to full-switch versus unchanged-context discrepancies. They are not task error rates. Each cell has its own normalization denominator; the raw absolute KL remains in the JSON. The changed wording affects both native context and propagated incoming e, so its effects cannot all be assigned to dc alone.

Nevertheless, context geometry independently shows the shift. Mean Euclidean context reconstruction errors are:

| Condition | Mean relative context reconstruction error |
|---|---:|
|Old wording / old edit|0.2178|
|Old wording / prefix|0.8662|
|New wording / old edit|0.5954|
|New wording / prefix|0.8748|

These means repeat each identical context over six signed-signal cases, with balanced weighting. The jump from about 0.218 in familiar conditions to 0.595 under new wording and 0.866–0.875 under prefix edits supports span coverage as a material limitation. A single nuisance operation such as letter→message is not represented by a fully wording-invariant direction in this frozen span. The fact that the old edit transfers imperfectly does not establish that no alternative invariant representation exists.

The losses are visible in competent copying and agreement, so they cannot be explained entirely by model task failure. However, the size of the wording effect differs strongly by task: sentiment's familiar edit degrades much less than copying's or agreement's. Task labels alone do not predict representational stability.

Updated hypothesis ordering:

1. Context coverage depends on both edit type and sentence construction. Strongly supported here.
2. Some edits have shared components plus wording-specific residuals. Plausible; the partial transfer and task variation motivate measuring the shared versus conditional response components directly.
3. The previous basis largely learned a small inventory of surface/context changes. Still plausible; H601's item transfer cannot rule this out.
4. Merely changing the coefficient metric can restore general transfer in this span. Weakened by H602's high oracle residuals and the present context geometry.
5. Every task needs unrelated computational units. Still unsupported: failure of one frozen span is not evidence against a better shared or conditional representation.

Next test whether new context coverage yields reusable edit-specific components. Train on several context constructions, hold out both task and wording where possible, and compare a fixed-rank shared span against condition-specific spans with matched total capacity. Keep output KL, answer margin, intended-token CE and internal response residuals separate. A stronger mechanistic result would identify a component that predicts interventions across constructions, not merely a basis that reconstructs more training differences.

Audits: no refit; frozen sources/maps/manifest unchanged; exact within-run unchanged-context logit replay; equal signal norms within 1.6e-5; oracle normal equations <=3.53e-15; finite metrics. Independent float64 evaluation matches all 960 saved cross terms within 4.11e-7 and verifies the finite-difference identity within 4.79e-13. The original-condition native prompts reproduce H600 within 2.39e-5 logits and 6.78e-7 relative MLP input error across 80 records, consistent with small batching-related numerical differences; aggregate relative metric differences from H601 reach 0.00310, so cross-run replay is not claimed exact. No new natural-text documents; FineWeb remains 26,112. Broad goal remains active and incomplete.
