## H780 — Key renaming changes the representations of unchanged copied-value words

Previous goal turn was progress: H779 localized answer-change effects to local values at copied-word positions, but its nuisance “other” category mixed unchanged value words with separators. H780 resolves that ambiguity using source roles shared across task and nuisance modes.

Token offsets from the original H777 prompts identify first and second value occurrences, key tokens excluding the final token, final query token, and remaining separator tokens. Tokenization is checked against every cached example. Value occurrences must each occupy one token, and the second occurrence must match the designated source position. Roles are disjoint/exhaustive and match between aligned receiver/donor prompts. The final token may be punctuation in colon/equals formats; “query” here means the final attention destination token, not necessarily the entire final key word.

With native receiver routing fixed, intervene on each role's donor-minus-receiver mixed value contribution, corresponding local-stream contributions, both value occurrences together, complete value edits and full head edits. No fitting or output-based selection occurs. The same480aligned edits and32nuisance alignment exclusions from H778/H779 are retained.

For key-renaming nuisance edits, the value words themselves are unchanged, yet their local values carry most of the value-vector change:

| Cohort | Complete nuisance value-effect RMS | Error using only both value-word positions | First-value isolated effect RMS | Second-value isolated effect RMS |
|---|---:|---:|---:|---:|
| Old keys / old values | .1592 | .0231 | .0455 | .1343 |
| New keys / old values | .1821 | .0462 | .0723 | .1725 |
| Old keys / new values | .1884 | .0422 | .0960 | .1624 |
| New keys / new values | .1769 | .0355 | .0503 | .1408 |

These are absolute answer-margin RMS quantities. Complete nuisance reference effects are below the established.25threshold, so relative effect-error percentages are suppressed. Errors above compare to the complete value-only edit, not to the full head edit, which also includes routing and interaction changes.

Across nuisance cohorts, the joint value-word vector has92.6–97.8% of the complete value-term RMS. Its mixed-value and local-stream contributions are equal up to numerical precision because the persistent layer-zero values at these unchanged tokens do not change. Non-query keys and final-query components are small by comparison; separators have nonzero residual contributions. Thus the earlier “other” result principally reflects contextual changes at unchanged copied-value words, not just separator effects or direct transport from the renamed key tokens.

For answer-changing task edits, both value-word positions reproduce H779's changed-word component exactly. Their error to the complete value effect remains9.3–17.5% on the novel cohorts. The second occurrence has larger isolated margin RMS in every task and nuisance cohort, but does not replace the first: second-only task errors to the complete value effect are44–50% on novel cohorts, versus9–18% using both. These isolated responses are not additive causal shares, and second-occurrence prominence is not proof of a universal recency mechanism.

This supports a concrete contextual-representation claim: before L5H5 reads the source values, upstream computation has made the representations of the value words depend on the key/context. It does not yet show whether this dependence encodes a reusable key–value binding, a superficial lexical feature, or some other contextual transformation. It also does not establish that the model needs this head to solve copying; these are matched intervention effects within a selected channel.

Audits:8,160 model forwards; roles verified for384examples; role vector partitions recombine within7.11e-15. Complete value/full-head effects replay H778 exactly; mixed/local role sums replay their respective reference margins exactly; identity logits replay exactly. Cached local/bus mixture error<=3.78e-6; outputs finite and hashes unchanged. Original reserved test untouched; natural document pool remains26,112. Source-role definitions and all per-example outcomes are saved.

Next: trace where the context dependence is created. Capture upstream attention/MLP outputs at the two copied-value positions in layers0–4, swap matched donor outputs one site at a time, and measure the resulting L5 local-value changes. Read those changes out through native receiver L5H5 routing in a fresh receiver forward pass, so downstream effects are isolated to the tested value channel rather than conflated with bypass paths. Include identity and full-target controls, task and nuisance contrasts, and lexical cohorts. Such a two-pass intervention is a path-specific diagnostic with its own assumptions, not a unique mediation fraction.
