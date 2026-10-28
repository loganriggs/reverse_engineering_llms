## H717 — Diverse task tests do not validate the provisional groups as selective units

Previous turn classification: progress. H716 showed a small conditional response alignment beyond exact decoder geometry, but uncertain context generalization. H717 tests predefined answer changes instead of output similarity.

Reuse the locked H533eight-family suite, with8discovery and8transfer pairs per family plus same-answer nuisance variants. This is a previously used research suite, not an untouched benchmark. Native competence requires at least4valid pairs, correct binary choice on both endpoints for at least75%of pairs, and at least50%pooled full-vocabulary semantic Top1. Evaluate all families, but causal eligibility depends only on discovery competence. No individual example filtering.

| Family | Discovery both-endpoint binary choice | Discovery full-vocab Top1 | Transfer full-vocab Top1 | Pass both splits |
|---|---:|---:|---:|---|
|agreement|100.0%|56.2%|56.2%|True|
|capital_recall|100.0%|87.5%|87.5%|True|
|context_association|75.0%|0.0%|0.0%|False|
|entity_binding|100.0%|100.0%|100.0%|True|
|negation|100.0%|37.5%|31.2%|False|
|relational_order|0.0%|0.0%|0.0%|False|
|sentiment|100.0%|62.5%|68.8%|True|
|sequence_copy|100.0%|81.2%|56.2%|True|

Five families pass: agreement, capital recall, entity binding, sentiment and sequence copy. Negation has strong binary discrimination but weak full-vocabulary answers, illustrating why contrast accuracy alone is insufficient. Relational order fails both measures. Context-association prompts have ambiguous intended completions and zero intended Top1; do not infer a mechanism for these failed tasks.

For every pair in the five passing families, test both task-swap directions and one nuisance-to-base swap. At block1final-token output, replace each group's decoded raw dual-coordinate contribution toward the donor, capped at5%receiver residual norm and the natural full group delta. Preserve the persistent value bus. Compare all8frozen groups, anchor-only contributions at the same group scale, and three substitute groups each including the same anchor. Substitute edits are rescaled to the real full-group edit norm. Anchor-only edits are not norm matched and can slightly exceed the full-group cap due cancellation; observed maximum6.17%.

The frozen selection score is mean donor-directed answer-margin change across the two task swaps, minus absolute answer-margin change under the nuisance swap. Select the largest discovery score, then evaluate that same group on transfer wording. This conservative score penalizes any nuisance sensitivity; nuisance and task edits can have different natural magnitudes below the cap and differ in input tokens/positions. It is not an unbiased estimate of a latent semantic quantity.

**Every candidate group has a negative discovery selection score in every passing family.** Therefore the selected rows below are least-negative candidates, not discovered positive mechanisms.

| Family | Selected group anchor | Discovery score | Transfer task-directed margin gain | Transfer nuisance absolute change | Transfer specificity score | Transfer95%interval |
|---|---:|---:|---:|---:|---:|---|
|agreement|121|-0.01476|-0.00028|0.02164|-0.02192|[-0.02808,-0.01598]|
|capital_recall|208|-0.00717|-0.00178|0.01221|-0.01399|[-0.02108,-0.00753]|
|entity_binding|216|-0.00277|-0.00126|0.00718|-0.00844|[-0.00981,-0.00716]|
|sentiment|121|-0.00052|0.00724|0.00707|0.00016|[-0.01239,0.01184]|
|sequence_copy|97|-0.02039|0.00055|0.03245|-0.03190|[-0.05242,-0.01439]|

Bootstrap intervals resample the8transfer pairs with both task directions and nuisance together,10000resamples. They are exploratory and not multiplicity-adjusted. Four selected families have clearly negative transfer scores. Sentiment is approximately zero with an interval spanning both signs. None establishes positive selective transfer.

The copy candidate beats the mean substitute control on the specificity score by0.02997, interval[0.01315,0.04775], but its own score is-0.03190and task-directed gain only0.00055. It is less disrupted by nuisance than those controls, not evidence of transferring a copy computation. Relative control superiority alone must not be relabeled success. All selected candidates leave receiver full-vocabulary Top1accuracy unchanged on transfer, and none yields donor Top1answers in these selected task swaps. Small edits need not flip strong answers, so lack of flips alone is not a rejection criterion.

The groups retain their earlier covariance and local response-alignment evidence. Those properties have not translated into convincing selective task effects. This does not rule out contextual or distributed computation; the tested site, undercomplete retained span, final-token-only edit and small-dose regime may be inadequate. In particular the persistent value bus is untouched.

Next run necessary positive controls before additional group fitting: full block1residual donor swap, its retained256Dprojection and complementary residual component, at the same small cap and at the natural full difference. Compare task and nuisance effects on the same competence-checked suite, preserving all negative results. This distinguishes a grouping failure from a site/dose/span limitation. If even a full residual swap fails, investigate where task information is available (including the value bus and other positions/layers) before expecting this early-layer featurizer to reveal the computation. If broader swaps succeed but groups fail, then change grouping/representation assumptions with an established causal target.

Audits:9984whole-model prefix evaluations,384native and9600causal; all128pairs token-valid. Control edit-norm match maxrelative2.63e-7. Independent CPUfloat64 recomputation of all1920group patch vectors agrees to2.24e-5relative; full-group norm cap verified. All model/source hashes unchanged and metrics finite. No original corpus reserved-test evaluation or new unique research documents; corpus pool26112. Broad goal active. All group results are retained, not only the discovery-selected rows.
