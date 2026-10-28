## H779 — Copying value edits reside mainly in local values at the copied-word positions

Previous goal turn was progress: H778 showed that receiver routing applied to donor-minus-receiver values approximated task edits much better than routing changes alone. H779 localizes that value term instead of assuming it is already pure word-identity transport.

The intervention destination is always the final L5H5 query. Partition the sources contributing to ArΔV into three disjoint sets: changed-token positions excluding the final query; final query; and all remaining positions. For task edits, the changed positions are the two copied-value words. For nuisance key renames, the changed positions are keys, so the meaning of the changed/other partition differs between modes. The query takes precedence when its token changes, making the partition disjoint and exhaustive.

Independently partition V into(1−lambda)local+lambda layer0bus, with lambda=−.0563965. Capture both streams from the actual model and evaluate the six crossed stream/position components as well. No representation is fitted. All results use the same480aligned edits as H778:256directed task edits,96nuisance edits and128identity edits;32nuisance alignment exclusions remain.

| Task cohort | Complete value-effect RMS | Local-only error to value effect | Changed-word-only error to value effect | Query-only effect RMS |
|---|---:|---:|---:|---:|
| Old keys / old values | .2336 | .0185 | .0308 | .0047 |
| New keys / old values | .2556 | .0185 | .0383 | .0061 |
| Old keys / new values | .3116 | .0180 | .0547 | .0050 |
| New keys / new values | .3036 | .0219 | .0281 | .0057 |

These errors compare with the **complete value-only intervention**, not with the full donor head edit. Across the three novel lexical cohorts, local-only error to that reference is5.8–7.3%, and changed-word-only error9.3–17.5%. Relative to the full head edit, changed-word-only errors remain29–40%, because routing changes and interaction terms are omitted. Keeping that distinction prevents inflating the result into near-complete model explanation.

Vector norms tell a consistent but non-attributional story: changed-word components have94–98% of the complete value-term RMS, query components about1.7–2.2%, and the persistent bus about7.3–8.7%. Local-stream component norms are slightly above100%, compatible with the signed bus contribution partially cancelling them. These are norm ratios, not percentages of causal responsibility. The layer-zero bus alone has small task margin RMS (.017–.020) and does not reproduce the full value effect.

Nuisance key renaming is different. The local stream almost exactly reproduces the value-only nuisance effect (absolute error.00055–.00101). But the dominant vector change lies in the “other” positions—tokens whose text is unchanged—rather than in the renamed keys or final query. “Other” nuisance components have about99–100% of the value-term RMS, and their absolute margin error to the complete nuisance value effect is.0176–.0464. The current partition does not distinguish copied-value words from separators within “other,” so it would be premature to identify which unchanged tokens carry that contextual signal.

This weakens a pure context-free word-identity interpretation: changing a key can change local values at unchanged positions. At the same time, answer changes are primarily carried by local values at the copied-word positions rather than by a representation already written at the final query. Together these results suggest inspecting context-conditioned source representations, rather than compressing all head outputs into a single global semantic subspace.

Audits:7,584 model forwards; local/bus mixture reconstructs captured mixed V within3.78e-6; position and stream partitions sum to the complete value vector within2.65e-6. H778 value/full-head reference margins replay exactly; partition sums replay value-effect margins within7.63e-6; native and identity logits replay exactly. All metrics finite and source/model/input hashes unchanged. Original reserved test untouched; natural document pool remains26,112. No claim of additive downstream effects or unique mediation fractions is made.

Next: refine source roles consistently across task and nuisance edits—first copied-value occurrence, second copied-value occurrence, non-query keys, final query and separators/remaining tokens. Verify roles form an exact partition and determine whether nuisance effects really arise at unchanged value words. Then trace how those source representations acquire key/context information upstream. This provides a concrete route toward understanding a reusable computation with contextual content, without claiming that the earlier failed rank-eight featurizers have become verified manifolds.
