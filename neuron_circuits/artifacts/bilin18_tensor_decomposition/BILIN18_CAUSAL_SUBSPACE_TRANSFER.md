## H995 — Causally selected task subspaces transfer and show selectivity, but routing and discovery remain unproven

Previous turn was progress: H994 scanned coordinate-level finite effects and froze six top8sets using discovery-only signed donor-response rankings. This turn applies ALLsix frozen subspaces to every validation family, rather than choosing an arm from evaluation labels. No basis refitting, new selection or featurizer training.

Each task-derived subspace has exactlyeight PCA directions. Independently project each endpoint, apply the joint donor-minus-recipient subspace change atMLP12, and compare against full donor-write replacement. Equal-rank controls: firsteight PCA directions and two seeded random eight-coordinate subsets. Dense128 and identity retained. Actual edit norms saved; equalrank does not imply equalnorm. Same312pairs/624directed edits as H991, including failed native diagnostics. Main summaries use50native-passing context cohorts and fourcopying/spatial holdout cohorts.

| Validation family | Own frozen subspace KL ratio | Own mean directed margin | Full-write mean margin | First8 PCA KL ratio |
|---|---:|---:|---:|---:|
|agreement|0.209|1.488|1.765|0.784|
|capital_recall|0.398|0.269|0.352|0.961|
|entity_binding|0.706|0.010|-0.039|0.937|
|negation|0.036|0.974|0.992|0.039|
|pronoun_gender|0.074|0.470|0.549|0.939|
|sentiment|0.058|0.915|0.935|0.068|

The full-distribution comparison is stronger evidence than the discovery sum of individual atom effects. Agreement, pronoun, sentiment and negation subspaces retain substantial selected-write behavior across new contexts. Capital recall also preserves some effect across the three competent validation cohorts despite its noisy discovery top1. Its own rank8 KL0.398 is better than the two random controls0.918/0.994, but coverage is limited to those three cohorts. Entity binding remains a weak-site diagnostic: own subspace pushes +0.010 while the fullwrite pushes-0.039. A positive selected subset is not a faithful account of the complete site's answer effect.

Cross-task response matrix shows selectivity: agreement's own mean1.488 versus0.006–0.168 for other source subspaces; capital's own0.269 versus0.006–0.047; pronoun's own0.470 versus0.006–0.157. Some cross-effects are appreciable and subspaces overlap; these are not independent mechanisms. The sentiment and negation subspaces largely transfer to each other: sentiment-source produces0.955 on negation versus negation-own0.974; negation-source produces0.903 on sentiment versus sentiment-own0.915. Shared answer groups and common valence output remain a confound for interpreting this as shared computation of negation. First8PCA directions already preserve these valence tasks well, limiting the incremental claim for supervised selection there.

Whole-task holdouts do not acquire useful responses from these supervised anchors: copying target-directed effects are onlyabout0.003–0.006 for most tasksubspaces versus fullwrite0.098; spatial effects at most~0.016 versus0.837. No evidence of discovering a new task emerges from these particular groups. Native competency of a holdout is not evidence that the anchoredMLP12 basis covers its mechanism.

Crucial implementation distinction: every subspace was supplied as a separate experimental arm. Evaluation task labels were used to summarize own/cross rows, not to choose an online gate. We have validated supervised candidate subspaces, not an activation-only routing policy, an unsupervised task classifier, a semantic manifold or a completed featurizer. The next step must integrate these anchors into actual gates rather than stop at the favorable own-task table.

Audits:7,488/7,488full-model forwards; identity logits2.48e-5; independent single-state projection difference agrees with direct difference projection1.53e-12. DensePCA128 control replays prior KL within2.54e-7; every cohort ratio independently reaggregated. Frozen selection/basis/model hashes unchanged, finite outputs, no reserved-test access. Natural-document pool26,112.

Next integration: use these frozen supervised sets as candidate overlapping dictionary groups, preserve background/unlabeled capacity, and compare activation-only gates against energy selection and explicit task-label oracle routing. Fit gates only on discovery states; apply without evaluation labels. Retain same atom budgets and union-once decoding. Preserve the candidate shared valence relationship rather than treating each task label as necessarily a distinct computational unit. Once routing and finite fidelity are validated, inspect unlabeled groups for new-task hypotheses and fresh validation. H961 context-bridge predictor evaluation remains pending; broad bootstrap goal stays open.
