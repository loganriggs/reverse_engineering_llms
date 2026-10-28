## H1005 — Anchor benefit survives random controls, but mixed geometry loses old causal contrasts

Previous turn was progress: H1004 established matched-rank whole-state gains from a supervised anchor in mixed geometry. H1005 adds two random-constraint controls and paired causal fidelity.10112forwards complete:256natural prefix references,5376whole-state state/arm evaluations,640full-donor references and3840paired candidate edits. No new tasks or natural documents.

Frozen task-only, mixed and mixed-real-anchor bases come from H1004. For seeds10051and10052, draw a Gaussian direction, project outside oldtask128 (as the true anchor lies there), normalize, and fit127leading mixed training directions orthogonal to that constraint. Every dictionary has128dimensions and the same mixed center where applicable. Random constraints use exactly the prior144task training slots/768natural discovery states; no controls or confirmation states fitted. Two random seeds are limited controls, not an exhaustive null distribution.

Whole-state results reproduce H1004. On40eligible new-control states, true-anchor KL0.026423 versus mixed0.034291 and random constraints0.034429/0.034187. Native-top1 agreement95% versus87.5%for mixed and bothrandomcontrols. Old and natural whole-state KL are nearly unchanged by the real constraint relative to mixed. Random constraints have slightly higher natural top1 agreement91.02%versus90.23%despite slightly worse KL; neither metric should be substituted for the other.

Paired edits add P(donorwrite-recipientwrite) to the native recipient. The full donor patch in that recipient context is the reference. Relative KL divides KL(reference || candidate) by KL(reference || native), pools within native-passing cohort, then macro-averages cohorts. Centers cancel for these linear differences. This is causal contrast fidelity, not single-state reconstruction or ground-truth task accuracy.

| Rank128 dictionary | Old-task paired relative KL,50cohorts | New-control paired relative KL,5cohorts |
|---|---:|---:|
| Task-only |0.0913|0.7741|
| Mixed |0.1936|0.5996|
| Mixed + real anchor |0.1951|0.3433|
| Mixed + random constraint1 |0.1940|0.6009|
| Mixed + random constraint2 |0.1941|0.5991|

The learned anchor reduces new-control paired error~43%against unconstrained mixed and bothrandomcontrols. This complements the whole-state improvement and supports a useful, narrowly scoped supervised contrast. It does not establish generic noun expectation or semantic manifold validity.

Mixed geometry more than doubles old-task paired error relative to task-only, even though old whole-state KL onlyrose~9%. By family, task-only→mixed-real-anchor relative paired KL: agreement0.114→0.135; capital recall0.208→0.545; entity binding0.210→0.506; sentiment0.026→0.034; negation0.010→0.016; pronoun gender0.020→0.114. Ratios depend on each cohort's reference effect and should not be read as equal absolute losses. The tradeoff is mostly caused by the mixed geometry, not by the added noun constraint. Plain reconstruction prioritizes directions differently from controlled computation.

Decision: do not replace the old geometry with mixed-anchor solely on whole-state KL. Keep both as baselines. The next bounded joint-supervision comparison should preserve the existing six-task discovery causal subspaces alongside the new anchor while filling the remaining rank128budget with mixed-data directions, using equally sized random constraints as controls. Existing task discovery interventions supply these anchors without adding validation labels. Then proceed to sparse allocation/group tests on the resulting candidates, carrying both fidelity metrics; do not keep tuning the noun case. This tests whether several supervised computation contrasts can prevent the old-task loss rather than assuming one successful anchor solves coverage.

Audits:10112/10112forwards; frozen sources/bases unchanged; no original reserved tests, no task-holdout fitting. Identity logits3.96e-5maximumerror; natural prefix capture relative3.78e-6. Independent36whole-state and24paired saved-logit samples replay exactly; all state and paired summaries/macros independently replayed; unique paired keys verified. Random direction generation and discovery-only constrained fits independently replay with projector error5.55e-16. Samples are not full independent replay of every logit. Natural-document pool26,112 unchanged. H961 context-bridge evaluation pending; broad goal active and sparse/manifold validation still incomplete.
