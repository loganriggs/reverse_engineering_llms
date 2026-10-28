## H923 — List-value promotion supports candidate names but does not explain the event-order preference

The preceding screen found second-event-name preference on explicit-fact QA despite failures to switch answers with the question. H923 retains all256 conditions, including failures, and tests each of the five name occurrences' persistent-value contributions at the three previously fixed readers. It also removes all five name contributions at one reader together. No readers or cases are selected using the new results.

Each ablation removes lamb times native signed routing times the layer-0 value for that occurrence, before the head output projection. Current-layer values, other positions and all other computation remain live. These are local zero ablations, not global stream deletions. Readouts include both names' logits/log probabilities, full-distribution KL, task accuracy and the event-second versus event-first margin independently of which answer was requested.

### Transfer results at L14H4

Baseline second-event-name preference is94.53%; mean name margin is0.620. Task accuracy is50.78%. The table aligns initial-list occurrences by the identity's later position in the event, rather than averaging over physical list position. Positive margin drop means reduced second-event-name preference.

|Removed contribution|Second-name logit change|First-name logit change|Second-name margin drop|Margin-change RMS|Second name preferred|Native→edited KL|
|---|---:|---:|---:|---:|---:|---:|
|identity|+0.000|+0.000|+0.000|0.000|94.53%|0.00000|
|list_event_first|+0.182|-1.086|-1.268|1.292|100.00%|0.07296|
|list_event_second|-1.033|+0.206|+1.239|1.262|6.25%|0.10379|
|event_first|-0.012|+0.048|+0.061|0.093|94.53%|0.00049|
|event_second|-0.043|+0.016|+0.059|0.081|92.97%|0.00043|
|extra_event_first|-0.016|+0.063|+0.078|0.085|95.31%|0.00041|
|extra_event_second|+0.073|-0.021|-0.095|0.104|95.31%|0.00080|
|all_names|-0.875|-0.789|+0.086|0.361|86.72%|0.08697|

Removing the list occurrence of the second event name reduces its logit by1.033 and drops its binary preference to6.25%. Removing the other list name lowers that other name's logit by1.086 and raises second-name preference to100%. Thus the initial-list values strongly support both candidate names. Their opposing effects cancel when averaged by physical list position: a near-zero mean margin change would misleadingly hide substantial individual effects. Identity-aligned means and RMS effects are therefore reported alongside positional results.

By contrast, removing the event-first or event-second occurrence yields only about0.06 mean margin drop and RMS changes below0.10. Removing all five name contributions at L14H4 lowers both name logits substantially but leaves86.72% second-name preference. This reader's name-value pathway therefore does not fully account for the event-order shortcut. The result does not establish strict independence: nonlinear compensation, current-layer values and other readers remain possible contributors.

L13H0 also promotes initial-list names, with smaller aggregate effects. Removing all its name contributions leaves94.53% second-name preference. L8H3 name-value effects are small on this format. All per-endpoint, per-query, discovery and transfer outcomes are retained rather than collapsing the analysis to successful task cells.

### Interpretation and next experiment

The evidence separates a reusable candidate-promotion operation from the operation that chooses between candidates. In the original sentence-completion panel, L14H4 also suppressed the repeated sender. In this explicit QA panel it mainly supports initial-list names, while much of the event-order preference survives removal of its local name-value inputs. A shared component can contribute to both competent behavior and a shortcut; its presence alone does not determine the behavior's meaning.

Next localize where the event-order preference arises. Use controlled event-argument swaps with names, list order, extra mention and question held fixed where possible; record native final-token residual/logit changes and test full attention/MLP donor substitutions across layers. Because an event swap can change the stated fact, score the event-order preference separately from task correctness and use matched semantic-preserving sent-to/received-from pairs as a second contrast. Start with native competence and token-alignment audits, then compare the two contrasts rather than treating all donor differences as one mechanism. Keep the earlier fixed readers as candidate-support controls instead of attributing the remaining preference to them by default.

This is localization toward a reusable computation model, not evidence that a manifold or an abstract binding feature has been verified.

### Verification

5,632 full-model forwards. Identity replay and native-cache replay both exactly zero. An independent masked-value contraction matches the occurrence-sum ablation delta to2.84e-14 max absolute error. Source hashes unchanged and measurements finite. Config, all endpoint metrics and identity-aligned summaries saved. No reserved-test evaluation or added natural documents; pool remains26,112. Broad goal active.
