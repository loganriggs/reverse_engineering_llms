## H884 — Context averages transfer response matching, but matching can conceal loss of native fidelity

Replaced exact lexical donors with partner averages estimated from other pairs, or with a single different-word donor. Task-context averages yield 4.2% language and 8.0% binding cross-context mismatch on the four pairs per domain excluded from estimation. Exact lexical identity is therefore not required for this response-alignment effect. However, agreement demonstrates that better matching can accompany substantial changes to the native computation. The next target must be native-response fidelity, not matching alone.

### Frozen estimation split and implementation

Within each domain, the first four sorted pair IDs are discovery and the last four are transfer. A context mean uses only the eight discovery endpoints within the same domain, family and template, at each of L8–17. Average the normalized raw pre-MLP vectors without renormalizing the mean: because bilinear partner activities are linear in that vector, this exactly averages the partner factors. Normalization's scale and radial correction remain recipient-native. Native recipient output offsets and input baselines are preserved, and attention remains live.

The different-word donor is deterministically selected from discovery by pair rank modulo four for transfer recipients, and the next discovery pair for discovery recipients. It preserves endpoint order. For agreement this also preserves grammatical number; it is a lexical mismatch, not a number mismatch. Compare live, recipient-native linearized, exact copy/task partner donors, discovery-mean copy/task donors, and different-word copy/task donors. All conditions retain identity and both write signs.

All transfer pairs were already measured in H883. They are excluded from mean estimation, but this is not globally untouched confirmation. The means are supervised by domain, family and template, and retain a full vector per layer. They are not an unsupervised featurizer, a demonstrated low-rank subspace, or a validated manifold.

### Transfer pairs: task-answer response mismatch under shared-scale uncapped readout

|Domain|Live|Exact copy donor|Mean copy donor|Different-word copy donor|Exact task donor|Mean task donor|Different-word task donor|
|---|---:|---:|---:|---:|---:|---:|---:|
|agreement|1.889|0.358|1.308|1.144|0.405|0.151|0.673|
|language|0.359|0.291|0.107|0.128|0.080|0.042|0.071|
|binding|0.240|0.361|0.156|0.165|0.038|0.080|0.083|

Task-mean absolute mismatch is 0.024 for agreement, 0.081 for language and 0.177 for binding, versus live 0.176, 0.610 and 0.546. Task-mean response norm ratios versus live are respectively (copy,task)=(0.90,1.72), (1.72,1.13), (1.17,0.97). Thus there is no uniform collapse to zero, but response amplitudes can change substantially.

Individual signed response mismatch under task means is 48.0% for agreement, 16.2% for language and 11.3% for binding. The corresponding exact-task-donor values are 27.1%, 18.0% and 11.4%. Agreement's improved canonical average conceals worse individual matching. Native and controlled readouts, discovery and transfer, source and task-answer contrasts remain separately saved.

### Native fidelity is a different criterion

A same-family exact donor recovers the recipient-native linearized transformation. Relative to that reference, same-family mean partner patterns change canonical task-answer responses by approximately:

|Domain|Copy-context response error using copy mean|Task-context response error using task mean|
|---|---:|---:|
|agreement|0.747|1.084|
|language|0.442|0.137|
|binding|0.336|0.040|

Agreement's task mean gives a response error exceeding 100% despite improved cross-context matching. Language and binding task means are substantially more faithful, but copying means still change task-answer responses by about 44% and 34%. A shared transformation can make two responses alike by changing both, rather than recovering what the native model computes. This is an experimental warning against optimizing cross-context alignment as the sole canonicalization objective.

Vocabulary-wide native-to-edited KL also remains distinct from fidelity to the live intervention. For binding task contexts, mean KL is 0.00520 live, 0.00614 with task means and 0.00900 with copy means. Language task contexts give 0.00197, 0.00320 and 0.00282. These measure disturbance from the unedited distribution, not approximation error against the corresponding live edited distribution. The latter should be an explicit next audit rather than inferred from these KL values.

### Next information-gaining test

Evaluate recipient-family context means as predictors of each context's own live intervention response. Add controls that use a mean pooled across copying/task contexts and an incorrect-context mean. Compare paired live-versus-approximated output distributions directly, alongside individual signed margin responses and absolute response strength. Use only discovery data to estimate any means or low-rank deviations, with transfer pairs excluded. Test whether retaining a small number of discovery-derived partner variation directions improves native fidelity on transfer pairs; do not select rank using transfer performance.

This puts the representation question back on a substantive criterion: can a compact description preserve the model's conditional computation, rather than merely manufacture invariance? A successful context average or small residual subspace would motivate a conditional featurizer; failure would indicate that lexical or other within-context structure remains necessary. Existing exact-donor alignment does not settle this.

### Verification and scope

4,800 full-model forwards. Identity final states and all H883 live/self/exact reference final states replay exactly. All ten layers retain independent autodiff, finite-difference and factorization checks. Selected readout error is at most 8.46e-06; no zero or tiny amplitudes were excluded. Independently verified 1,536 donor assignments, all discovery-only mean supports, and lexical separation for transfer wrong donors. Identity KL max absolute 8.73e-12. Source hashes unchanged and outputs finite. Four transfer pairs per domain give limited generalization evidence; no population-level confidence claim is made. No reserved test evaluated, natural-document pool unchanged at26,112. Broad goal active.
