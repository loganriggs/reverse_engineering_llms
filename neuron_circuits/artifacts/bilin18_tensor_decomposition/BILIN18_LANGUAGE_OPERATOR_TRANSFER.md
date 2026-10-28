## H910 — The frozen L13 language-context map transfers to new countries; L14 is substantially weaker

The previous turn made progress: H909 froze the evaluation panel and competence strata before intervention outcomes. H910 evaluates unchanged H908 task-context means, plus copying-context means averaged from their frozen A/B means, on every eligible row at L13/14. Each map is applied independently to every family/template; no runtime family-based selection or fitting occurs. The copying pooled control is a fixed arithmetic combination of existing discovery means, not a fit to the new panel.

All224 rows are retained: old-country A/B are reused controls, old C/D are changed prompts, new-country A/B/D are competent transfer cells, and new-country C is a failed-gate diagnostic. Exact, identity, native-linear and quadratic-only controls are included, as are mean-plus-quadratic variants. The evaluated change remains a final-query single-MLP matched-source intervention, not a whole-prompt swap.

### Competent new-country task transfer

The task midpoint approximation is J(mean_task)d, keeping the full country-specific change d. Margin-error ratio is RMSE relative to the exact intervention divided by the error from doing nothing. KL ratio uses KL(exact-edited || approximation) divided by KL(exact-edited || native), so the reference direction is consistent. These ratios are not fractions of causal computation explained.

|Layer|Template|Task-map margin-error ratio|Task-map KL/no-change|Copy-map margin-error ratio|Copy-map KL/no-change|
|---|---|---:|---:|---:|---:|
|13|A|0.365|0.174|0.757|0.767|
|13|B|0.331|0.153|0.777|0.832|
|13|D|0.344|0.194|0.744|0.841|
|13|competent|0.346|0.172|0.761|0.814|
|14|A|0.780|0.451|0.879|1.000|
|14|B|0.825|0.435|0.890|1.021|
|14|D|0.763|0.473|0.860|1.038|
|14|competent|0.792|0.454|0.877|1.021|

At L13, the frozen task map retains similar target-margin fidelity across A/B/D, including new countries in new template D. Competent aggregation (36 task endpoints across six pairs and three templates) gives margin-error ratio0.346 and KL ratio0.172, with absolute KL0.00672. The exact intervention's RMS target-margin effect is0.835 logits. The copying-context control is substantially worse (0.761 margin-error ratio,0.814 KL ratio).

L14 does not replicate that strength: competent margin-error ratio0.792 and KL ratio0.454, with full-effect RMS0.441. It improves distributional fidelity relative to doing nothing, but reproduces target-margin changes poorly. Preserve this failure rather than generalizing the L13 result to all late language MLPs.

New-country C remains a failed-gate task condition despite correct binary preferences. Its L13 task-map margin/KL ratios are0.342/0.387; L14 gives0.753/0.778. These are diagnostic results only and do not upgrade its task competence.

### Context specificity and quadratic controls

On copying rows matched to competent new-country task cells, L13's language map is worse than doing nothing: margin-error ratio1.365 and KL ratio2.306. The copying map improves those to0.541/0.421. At L14 the corresponding language-map ratios are1.595/2.726 versus copying-map0.704/0.333. Copy prompts repeat across some template labels; these summaries use the matched cell weights and are not independent-sample confidence estimates.

Adding Q(d) to the task mean is not a universal improvement. For competent new-country tasks, L13 task-mean-plus-quadratic yields margin/KL ratios0.389/0.643 versus midpoint-only0.346/0.172. At L14 it yields0.810/2.411 versus0.792/0.454. This supports treating the mean as an approximation to the pair midpoint, consistent with the exact bilinear identity, rather than indiscriminately adding a quadratic correction.

Old-country aggregate task-map margin ratios remain0.327/0.367 at L13/14. These old aggregates mix reused A/B with changed C/D prompts and include original discovery countries; they are controls, not independent transfer evidence. The L14 deterioration on new countries is therefore a material generalization limitation.

### What the result supports

There is now prospective intervention evidence for a frozen L13 context-dependent map processing additional lexical directions, including a new prompt variant, with a meaningful wrong-context control. This is stronger evidence for reusable processing than the previous small-subspace reconstruction results. It remains an approximate operation built from original model weights, a supervised discovery context, full donor changes and selected layers. It is not an automatically discovered dictionary, low-dimensional manifold, or complete language circuit, and it does not establish task necessity or answer-switch sufficiency.

Next test stability of the operator across disjoint discovery country sets: derive additional means using earlier old transfer countries, compare the resulting maps and causal effects, and label any re-evaluation on the now-used H909 panel as diagnostic. Do not refit on new-country outcomes and claim fresh confirmation. Binding pathways and shared-dictionary/featurizer validation remain outstanding. Broad goal active.

### Verification

4,032 full-model forwards. Identity, cached normalized activation replay and baseline-logit replay against H909 are all exactly zero error in this execution. All source/model/frozen-mean hashes unchanged and all recorded KL finite. Every eligible row and predefined competence stratum retained. No natural or reserved-test evaluation; document pool remains26,112.
