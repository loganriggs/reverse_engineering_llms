## H848 — One task-specific direction preserves most of the relay's causal effect across vocabulary/format

**Main result.** Fit on only8 discovery contrast pairs/task, a single preMLP4 direction preserves83.5% of agreement's full relay-swap mean effect and87.1% of sentiment's on the basis-held-out vocabulary/format split. Individual-effect relative RMSE is0.171/0.250. The other task's basis and matched-norm random controls preserve little. This is stronger evidence for a compact task-specific contextual direction than reconstruction alone, but it is supervised contrast compression and not yet independent confirmation.

**Fit and evaluation.** Uncentered SVD of the eight discovery raw-state donor-minus-recipient differences at the same-token relay site. Agreement uses near/beside; sentiment the period. One orientation per pair avoids double-counting reciprocal vectors. Freeze ranks1/2/4/8 before causal evaluation. Project actual donor contrasts onto each basis, add to the native recipient state, and leave the rest of the model native. This uses donor activations to obtain coefficients, not a task-label-only predictor. The orthogonal complement is tested separately. Random baselines include three fixed subspaces and three directions scaled to the own-projection norm, with both signs. Other-task projection uses the same site/coordinate dimension.

Transfer examples are held out from fitting, but H847 used them to choose layer/position. Thus this is not a fresh research holdout. Agreement has8 transfer pairs; sentiment7 after earlier alignment exclusion. Repeated directions/signs are not independent examples.

|Task|Edit|Rank|Mean full relay effect preserved|Individual effect relative RMSE|
|---|---|---:|---:|---:|
|agreement|own|1|83.5%|0.171|
|agreement|complement|1|1.7%|0.988|
|agreement|other|1|-0.1%|1.001|
|agreement|random_matched|1|1.7%|0.986|
|agreement|own|8|83.4%|0.174|
|agreement|complement|8|2.2%|0.989|
|agreement|other|8|0.2%|0.999|
|agreement|random_matched|8|1.9%|0.985|
|sentiment|own|1|87.1%|0.250|
|sentiment|complement|1|4.5%|0.954|
|sentiment|other|1|0.1%|1.000|
|sentiment|random_matched|1|3.1%|0.979|
|sentiment|own|8|91.1%|0.196|
|sentiment|complement|8|2.3%|0.975|
|sentiment|other|8|-0.0%|0.999|
|sentiment|random_matched|8|3.5%|0.977|

Full-effect preservation is a ratio of aggregate signed answer-margin changes, not accuracy or unique mediation. Individual effect error prevents interpreting the ratio alone as faithful prediction. Full output KL(full-swap||approximation) is also retained; causal answer fidelity and whole-output fidelity remain different objectives.

**Geometry versus function.** On transfer, rank1 retains only45.8% of agreement contrast energy and59.1% of sentiment contrast energy, despite preserving much more of the causal answer effect. Rank8 retains50.6%/65.6%; it does not materially improve agreement's effect error, while sentiment improves to0.196. The discarded vector remainder has little answer effect in isolation. This is compatible with a dominant causal direction embedded in larger lexical/context variation. It does not prove that remainder is irrelevant to other predictions or that effects add: nonlinear downstream computation can create interactions.

The two tasks' leading directions have squared cosine0.00102. Cross-task projection has negligible answer transfer. This weakens a single universal relay direction and supports task-specific content carried through a possibly shared positional mechanism. It does not imply independent neurons, disjoint circuits, or a verified semantic taxonomy.

**Next confirmation.** H849 freezes these bases and original relay choices on new source-word pairs, two fixed formats and three gap variants. It tests native competence before interpretation and preserves failed/unaligned cells. Gaps alter wording as well as distance, so a change cannot be attributed to distance alone. No outcome-driven basis refit or movement to a newly favorable relay position is allowed. The more difficult question is whether the direction remains useful when lexical content is no longer immediately adjacent to the relay.

**Audits.**3,100 full-model forwards; basis orthogonality error1.45e-15, identity logit error4.20e-5, full donor effects reproduceH847 within5.73e-6. Sources/checkpoints unchanged; finite metrics; no reserved-test evaluation and natural-document count unchanged at26,112. Broad goal remains active; this is a promising scalar causal candidate, not completion of manifold/alternative-featurizer validation.
