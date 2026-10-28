## H845 — Causal content-site reconstruction fails, but unseen-token leakage changes the interpretation

**Main result.** On early content positions with a measurable causal route, all current frozen featurizers lose much of the exact mean-centered donor difference. However, the largest apparently contextual effects depend strongly on tokens missing from the training token-mean table. The evidence supports poor transfer of these representations to this task panel, not a clean conclusion that they fail to encode purely contextual task features.

**Prior evidence audit.** H537 already showed that much of binding/copying's bus effect survives fixed recipient-native Q/K readers (97.6%/99.7% of unrestricted mean effects on its transfer panel). H739 already factorized L13H0's repeated-token substitutions into a fixed value/output map times a signed contextual coefficient, with absent-token tests and explicit format confounds. H778 already separated routing/value/interaction edits for L5 copying. H844 therefore extends a known persistent-value explanation to the current suite and complete-state factorial; it must not be advertised as the first discovery of that route. Further bus mapping should address remaining operation-versus-format generalization, rather than rerun generic localization.

**Design.**31 equal-length pairs:16 agreement,15 sentiment, two directions, sites1/2. Capture raw preMLP states at all positions and patch only token-differing positions before both MLP and residual skip. Keep recipient x0 and persistent values unchanged. This matches the actual input site of H836 featurizers, unlike applying their decoder to a block-output state.

Define normalized shape u, frozen training-token mean M, remainder C=u-M, and raw RMS radius. Run all8 donor/recipient combinations of M,C,radius, renormalizing each mixed shape and restoring the selected radius. C is a mean-centered remainder, not a demonstrated pure-context variable. For each of five H836 architectures and both seeds, replace exact C donor-minus-recipient with decoded reconstruction difference, preserving recipient reconstruction error. Evaluate C alone and M+C at recipient radius. No training, task-driven feature selection, or checkpoint changes.

**Aggregate transfer.** Values are percentages of the native donor-recipient answer-margin gap, not accuracy or unique causal attribution. Learned-model columns average both seeds. Radius-only effects are near zero; full-factor and direct raw donor replacement agree exactly in answer-margin effects.

|Site|Task|Exact M only|Exact C only|Exact M+C|Scalar64 C|Block8 C|Learned shared C|
|---|---|---:|---:|---:|---:|---:|---:|
|1|agreement|19.6%|21.8%|71.5%|0.3%|0.3%|0.1%|
|1|sentiment|18.2%|28.0%|58.8%|0.4%|-0.1%|-0.1%|
|2|agreement|26.8%|16.8%|64.2%|1.1%|0.7%|0.2%|
|2|sentiment|14.1%|31.7%|59.0%|5.0%|0.9%|-0.7%|

The frozen models also miss the geometry of the changed-token C differences. For example atL2, scalar64 contrast relative RMSE is about0.77–0.78 on agreement and0.82 on sentiment; fixed shared groups are about1.00 and0.97–1.00. These compare actual donor-minus-recipient vectors, not single-state reconstruction scores. C-only task-effect relative RMSE ranges roughly0.87–1.01 across these models/cells. M+C effects improve mainly because the independently supplied token mean carries additional signal; this does not establish that the learned contextual coordinates are faithful.

**Critical coverage audit.** Only8/16 agreement pairs and3/15 sentiment pairs have both changed-token identities represented in the training mean table. Other tokens use the global fallback. Their C therefore retains substantial token-specific signal, so pooled C effects cannot be called contextual computation. All rows remain retained; the following is a posthoc diagnostic, not a newly selected evaluation panel:

- On the8 fully covered agreement pairs, exact C transfers5.57%/4.76% atL1/L2. Scalar64 transfers0.87%/0.50%; the other learned models also transfer less on average. A remaining discrepancy exists, but it is much smaller than the pooled effect.
- On only3 fully covered sentiment pairs, exact C transfers1.02%/0.19%. Learned models produce similarly small, sometimes larger effects. This tiny subset does not support a meaningful architecture ranking or strong contextual-feature failure claim.

Even a present token mean can be noisy or context-biased, and different token identities remain a confound. Seen-token stratification reduces the global-fallback issue but does not create a same-token contextual test.

**What changes next.** Do not label the shortlisted components as general task manifolds or train a hierarchy around them yet. Two live explanations must be separated: (1) inadequate lexical coverage/centering produces out-of-distribution contrasts, and (2) reconstruction training discards low-variance but causally useful context directions. A useful next diagnostic holds the token identity and position fixed while varying upstream context, and first establishes a whole-state causal effect at that same-token site. In parallel, construct a coverage-controlled panel prospectively from training-supported tokens, retaining diverse formats and native competence checks; do not retrospectively pick the strongest8 cases. If a contextual route exists, compare exact remainder, fixed subspace reconstruction and learned coordinates at matched dimensional budget before adding richer graph assumptions. Prioritize independent confirmation after choices are frozen.

**Audits.**3,720 whole-model forwards; native runner exact; reconstructed-identity logit error<=4.20e-5; full-factor versus raw donor margin error0; frozen sources/checkpoints unchanged; all metrics finite.147 unknown token occurrences/site across all captured positions, with changed-position coverage separately saved by pair. Original reserved test untouched and natural-document pool unchanged at26,112. The previous turn made progress by localizing causal routes; this turn tests representations at the correct causal site and exposes a consequential lexical-coverage confound. Broad goal remains active.
