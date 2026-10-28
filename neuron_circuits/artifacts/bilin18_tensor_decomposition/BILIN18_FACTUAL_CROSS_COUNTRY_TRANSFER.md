## H1049 — Factual-task transfer collapses when country identity is excluded

Previous goal turn was progress: H1048 completed30whole-family fits and found gains concentrated in capital/language tasks sharing country content. This turn evaluated the frozen factual crossfit models on fixed matched-wording seen-versus-excluded-country panels. No fitting, rank search, output-based support selection, or cohort filtering.

Each task/cohort has fourpairs. Seen cohort uses France/Germany,Italy/Japan,Spain/China,Russia/France: seven unique countries, with France repeated, all present in the corresponding fold's training prompts through the other factual task. Excluded cohort uses Poland/Turkey,Finland/Sweden,Norway/Portugal,Denmark/Hungary: eight countries verified absent by whole-word matching from every actual prompt used to fit the fold. These countries were historically evaluated elsewhere, so exclusion concerns current representation fitting, not pretraining or globally fresh research data. Repeated France and differing identity counts limit independence.

Wording is fixed to the original task templates: capital city versus main language. Both endpoints have equal token lengths, disjoint single-token answer groups, and preserved continuation boundaries. Native competence passes for all four cohorts: both-endpoint accuracy100%; top1capital seen/excluded100%/100%, language87.5%/100%. Thus poor excluded-country edits are not explained by native task incompetence on this panel.

Paired relative KL, lower better:

| Task/cohort | Scalar initial→trained | Block initial→trained | Shared initial→trained | Dense projection |
|---|---:|---:|---:|---:|
| Capital, seen countries |0.1002→0.0525|0.3718→0.1741|0.6292→0.3508|0.0150|
| Capital, excluded countries |0.8125→0.7994|0.8977→0.8995|0.9144→0.9083|0.6236|
| Language, seen countries |0.1038→0.0213|0.3812→0.1137|0.4635→0.1181|0.0074|
| Language, excluded countries |0.7830→0.7778|0.8806→0.8720|0.8867→0.8775|0.6090|

Trained scalar donor-answer top1is50%capital/100%language for seen countries, versus0%for both excluded cohorts. Every candidate, including dense projection, has0%donor top1on excluded countries. Dense uses384coordinates instead of sparse max48 and is a fixed Euclidean reference, not an information-theoretic upper bound on every encoder using the span.

This sharply limits H1048's apparent task-level reuse: it largely transfers across factual readers when lexical country identities are shared. It does not demonstrate a content-independent factual operation. Since even the untrained dense projection changes from near-faithful seen edits to poor excluded edits, gate optimization alone cannot explain the gap; the fitted input geometry already loses relevant new-content variation. The experiment does not uniquely locate that loss to a single port or prove irrecoverability from all possible codes. Learned decoder behavior, low-rank coverage, and content distribution shift remain intertwined.

Connection to earlier research: the native model already reuses token-derived cached values across later readers, and earlier weight-based lexical payload work transferred between language and capital contexts. These findings are consistent with reusable computation acting on content-specific vectors rather than one small global activation dictionary covering every entity. This is a hypothesis supported by converging evidence, not a newly established complete mechanism.

Verification:32prefix captures/256suffix calls, no fits, all original crossfit geometry/checkpoint hashes unchanged, panel hash unchanged. Excluded-country mentions in fitting prompts0; seen mentions verified for each fold. Independently recomputed KL for28saved logit samples, max2.40e-14. All cohorts and candidate outcomes retained. No new corpus documents, externally fresh task validation, or manifold discovery claim.

Stopping criterion met for the factual lexical-transfer test. Do not extend into another head-localization campaign. Next representation-design experiment should explicitly separate token-specific content from contextual changes: build a content-preserving baseline using the existing exact token-to-cached-value program, with any native/uncompressed bypass counted and disclosed, then apply sparse action dictionaries to the remaining contextual streams. Give every compared architecture identical content access. Validate the extracted content program numerically on diverse existing tokens before interpreting behavioral effects. This tests a conditional representation contract rather than assuming broader supervision can make a small fixed PCA span represent all entities. It remains necessary to test genuinely reusable components and unfamiliar task combinations before claiming the broader research objective is resolved.
