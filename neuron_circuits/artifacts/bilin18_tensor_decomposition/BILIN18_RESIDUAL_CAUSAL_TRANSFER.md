## H1000 — New behaviors depend on the old task span's residual, but not its cluster-center span

H999 was progress: it completed and independently audited four residual-inspired behavioral screens. H1000 now runs640finite interventions at the fixed joint-featurizer site MLP12, using only noun_expectation and word_fragment. List markers remain quarantined near the copying holdout; clause completion remains diagnostic. No featurizer fitting or reserved-test evaluation.

For each of32pairs, patch both directions. Decompose the donor-minus-recipient MLP write into the frozen H991 task PCA128 projection and its orthogonal residual. Compare identity, full donor, task128, full residual, two rank8spans of H998 natural-discovery cluster centers, two seeded random rank8residual spans, and paired positive/negative residual noise norm-matched to the first cluster-span edit. The cluster centers are projected orthogonal to task PCA before SVD, with all8singular directions retained. No task outcome selects a span, layer or example. These edits use donor differences; they are not deployable single-state predictors.

Results below aggregate the24directed interventions across lexical, structural and joint cohorts for each family. Signed answer-margin effect is relative to the full-donor effect within the same recipient context. Relative KL is KL(full-donor-patched distribution || component-patched distribution), divided by the corresponding identity baseline, using ratios of aggregate means. Lower KL is better; zero means matching the full-donor output. These two metrics are distinct.

| Intervention | Noun effect/full | Fragment effect/full | Noun relative KL | Fragment relative KL |
|---|---:|---:|---:|---:|
| Old task PCA128 |0.182|0.194|0.793|0.851|
| Full orthogonal residual |0.793|0.799|0.094|0.076|
| Natural cluster-center span,seed9981 |-0.0004|-0.014|0.969|0.999|
| Natural cluster-center span,seed9982 |0.014|-0.0006|0.975|0.988|
| Random residual rank8,seed10001 |0.016|0.019|0.988|1.013|
| Random residual rank8,seed10002 |0.010|-0.004|0.999|0.970|

Full-donor mean answer-margin effects are0.924 for nouns and0.722 for fragments. Discovery results agree qualitatively: strong full-residual effects, weak old-span effects, and nearly absent cluster-span effects. Norm-matched positive/negative noise has small effects; this is a limited random control, not proof against all off-distribution artifacts. Ratios of finite effects are not an additive causal partition because downstream computation is nonlinear.

Interpretation: these probes expose behavior-relevant directions missing from the six-task learned PCA geometry. Natural residual clusters helped propose probes, but their center spans do not encode the corresponding donor effects. This does not rule out meaningful structure within clusters, continuous manifolds, other sites or better discovery methods. The full residual is1024dimensional, far larger than either rank8control; its success establishes missing behavior-relevant variation, not compact reusable units.

Confounds remain: noun probes change instrument and action together; subject splits reuse actions. Fragment probes vary definitions and lengths and test only one BPE continuation. Same-final-token matching cannot remove those confounds. We have causal effects on the tested contrasts, not a validated abstract noun or morphology module.

Next information-gaining comparison: use only four discovery pair differences per new family to learn small residual subspaces, then freeze and test all lexical/structural/joint cohorts and cross-family transfer against rank-matched random subspaces. Compare pooled versus separate task spans and an augmented old-task representation. This distinguishes shared transferable missing directions from source-specific memorization. Before claiming semantic labels or retraining the broader featurizer, add factorial instrument/action and fragment cue controls. Preserve the ultimate goal of improving the featurizer and validating new-task discovery, rather than another extended single-task detour.

Audits:640/640distinct interventions; identity logit maximum error2.57e-5; task+residual decomposition error2.84e-14; basis orthogonality error9.99e-16; noise norm mismatch2.27e-13. Independent saved-logit replay exactly reproduces all margin effects and reference KLs; independent projection-norm replay error0. All hashes unchanged, finite metrics, no fitting or reserved tests. Natural-document pool remains26,112. H961 context-bridge evaluation remains pending; broad goal active.
