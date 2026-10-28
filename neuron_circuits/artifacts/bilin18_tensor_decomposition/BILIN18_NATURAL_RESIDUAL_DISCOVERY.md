## H998 — Natural-text pilot reveals a large task-span residual and candidate continuation phenomena

Previous turn was progress: H997 completed the allocation control and ended the narrow architecture sweep. This turn begins the actual unlabeled discovery branch using existing permitted natural-text data, not more task-label tuning.

Read the H681 document manifest and select256documents ONLY from its explicit2048-document training split, by deterministic H998 salted document-hash order. No H681 validation/test document selected. Retrieve only selected source-array rows and verify their saved128token-window hashes. First192documents form pilot discovery, remaining64 pilot confirmation; document hashes are disjoint. These are reused research documents, not new corpus draws or model-training holdouts. Underlying arrays/manifests are hashed for provenance; only selected train rows are evaluated.

Capture MLP12 outputs at predetermined positions31,63,95,119 (1024states) and native next-token diagnostics. Entire128-token forward uses causal attention; one independently truncated32-token forward audits that future tokens do not change the first sampled output. Task-trained PCA128 basis/center from H991 stays frozen. Compute its orthogonal residual; fit eight spherical clusters to768discovery residual states only with two seeds and50fixed iterations. Assign256confirmation states without refitting. Exemplars are selected from discovery only, deduplicating documents within each cluster card. Native next-token labels are recorded for interpretation, not used in clustering.

| Pilot split | Task PCA128 relative squared reconstruction error | Scalar32 error | Native exact-next-token top1 |
|---|---:|---:|---:|
| Discovery,192documents |0.638|0.686|0.402|
| Confirmation,64documents |0.646|0.695|0.430|

Errors are relative to x minus the frozen TASK training mean; they include distribution/mean shift and are not variance explained around the natural-text mean. Native exact-next-token rates are ordinary ambiguous language-model continuation statistics, not task competence gates. This large unrepresented natural variation qualifies earlier strong synthetic-task results: the task-trained span is not a faithful general natural-text representation. No natural-text downstream reconstruction interventions have yet been run.

Seed9981 prototype inspection suggests: cluster2 sentence/newline boundaries; cluster3 completed-clause punctuation readiness; cluster7 determiner/noun-phrase continuations; cluster5 partial-word/name/heading continuations; cluster4 prepositional arguments; cluster0 connectives. Cluster1 mixes a medical list with other contexts, and cluster6 is heterogeneous. These are tentative observational descriptions, not labels assigned to every member. Period is40% of cluster2's current tokens; several other clusters have modal-token frequencies below15%. Token confounding remains substantial and unequal.

Two-seed confirmation co-membership-pair Jaccard0.541. On the54confirmation states whose current token occurs at leastfourtimes, observed Jaccard0.445 versus mean0.264/max0.346 under64within-exact-token shuffles of the second seed's assignments. This limited descriptive control suggests some shared contextual structure beyond token identity, but is not a significance or causal-unit test. Four states/document and only64confirmation documents limit independence; the8cluster choice itself imposes a discretization.

Twenty hypotheses registered before follow-up behavioral tests (full evidence/discriminators saved in JSON):
1. Sentence restart after boundary: Match current boundary token; vary whether sentence ended; measure next-token start distribution and residual edits.
2. Clause-completion punctuation readiness: Same final lexical token in complete versus incomplete clause; test punctuation probability and causal swaps.
3. Determiner-to-noun continuation: Hold final determiner fixed across predicted noun categories; compare residual versus token-only effects.
4. Within-word continuation: Separate token-internal continuation from new-word boundaries, matching current BPE token where possible.
5. Coordination and list continuation: Parallel items versus completed list with same conjunction; validate on several semantic domains.
6. Prepositional argument completion: Match final preposition and manipulate expected argument type; use neutral-topic controls.
7. Discourse connective continuation: Compare connective uses with same token in other constructions; test completion distributions.
8. Entity-category expectation: Hold formatting and current token fixed while changing entity-category cues.
9. Domain-specific vocabulary state: Cross topic and syntax factorially; test whether effects follow domain or syntactic position.
10. Heading/list formatting state: Compare identical content in prose, headings and lists; examine continuation format.
11. Delimiter closure state: Fresh matched quotation/bracket-open versus closed contexts, with length controls.
12. Local morphology constraints: Use token-valid inflection/derivation pairs with lexical and boundary controls; avoid existing tense holdout training.
13. Prediction uncertainty: Stratify by native entropy and test whether cluster patterns persist within strata.
14. Unexpected-continuation state: Compare continuation surprisal at matched token/position; avoid using future tokens for fitting.
15. Current-token identity artifact: Exact-token matched controls and token-conditioned subtraction on discovery data only.
16. Absolute position artifact: Report cluster occupancy by position and replicate at new predeclared positions.
17. Residual-norm/outlier artifact: Norm-matched strata and robust removal of predeclared extreme-norm states.
18. Document/style/boilerplate artifact: Document-block resampling and new-document replication; count independent documents.
19. Task-span domain shift: Compare train-only natural PCA and task PCA on separate pilot documents with downstream fidelity.
20. Clusters are quantization, not natural units: Compare continuous subspace prediction and multiple cluster counts on discovery only; require causal selectivity before naming units.

Prioritize sentence restart/clause completion, noun-phrase continuation and within-word continuation, with current-token, position and document controls. Do not retrofit strong semantic names to these clusterings. The next deliverable should be fresh token-valid behavioral contrasts plus finite residual-component interventions; proposals that merely reproduce token identity or fail native competence should not enter retraining.

Audits:257full-model forwards (256windows,oneprefix replay). Prefix logits agree3.34e-5; captured MLP12 relative error1.61e-6. Source/manifest hashes unchanged; residual orthogonality9.32e-16 relative norm and independent residual replay1.14e-13 absolute. Bothcluster assignments independently reconstructed from frozen centers. Original nontrain documents used0, no original reserved-test evaluation, no task-holdout fitting. Natural-document pool remains26,112 because all documents reused.

The bootstrap loop now has a concrete unlabeled sample, residual patterns and testable candidate hypotheses. It has not yet discovered a validated new task or improved a featurizer by retraining on one. H961 context-bridge predictor evaluation remains pending; broad goal stays active.
