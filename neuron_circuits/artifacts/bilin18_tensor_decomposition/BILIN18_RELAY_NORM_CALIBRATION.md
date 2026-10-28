## H856 — Task-direction-free contrast norm restoration rescues much of active-span fidelity

**Main result.** Restore the original donor contrast's norm after projecting into the frozen active decoder span, without loading a task direction. This improves answer-effect fidelity and full-output KL relative to ordinary projection in every one of the60 model/task/gap aggregate cells tested. Much of the previous oracle deficit is attenuation rather than complete absence of a useful direction. Learnedshared models still retain substantial deficits.

**Rule and scope.** For true rawpreMLP4 contrast delta and projector P onto the union of the two native active atom sets:

`p = P delta`

`norm_restored = p * ||delta|| / ||p||`.

This holds the projected direction and support fixed and restores exactly the original contrast magnitude. No task-direction file, task label or task effect enters this formula. Also test fixed gain2 as a separate exploratory control, specified before this run. All10 fits, all78 H849 pairs, both directions and all gap variants retained. These are raw-state edits with no postpatch normalization, matching H855 rather than H854's shape/radius protocol.

**Adjacent-prompt examples.** Preservation is relative to the full raw relay swap; values above100% are overshoot, not superior accuracy. Individual-effect relativeRMSE is the more direct fidelity measure.

|Task|Model|Ordinary projection preserves|Norm restoration preserves|Effect relative RMSE|
|---|---|---:|---:|---|
|agreement|scalar48_8351|50.2%|96.5%|0.513→0.094|
|agreement|scalar48_8352|61.3%|102.1%|0.395→0.059|
|agreement|scalar64_8351|50.0%|88.1%|0.517→0.176|
|agreement|block8_8351|57.1%|97.2%|0.463→0.129|
|agreement|shared_fixed_8351|29.6%|74.6%|0.712→0.309|
|agreement|shared_learned_8351|32.7%|77.8%|0.684→0.250|
|agreement|shared_learned_8352|15.4%|54.9%|0.852→0.488|
|sentiment|scalar48_8351|59.6%|91.0%|0.423→0.153|
|sentiment|scalar48_8352|55.2%|84.4%|0.482→0.310|
|sentiment|scalar64_8351|47.2%|75.5%|0.544→0.289|
|sentiment|block8_8351|60.7%|91.2%|0.417→0.188|
|sentiment|shared_fixed_8351|46.2%|77.6%|0.564→0.400|
|sentiment|shared_learned_8351|19.5%|48.0%|0.802→0.536|
|sentiment|shared_learned_8352|9.3%|29.4%|0.907→0.721|

Across all three gap variants, norm restoration improves individual answer-effect error in all60 aggregate cells (10 models×2 tasks×3 gaps). It also lowers KL(full-swap output||approximate output) in all60 cells. Relative KL reductions are strongest on adjacent agreement (ratios .097–.415 of ordinary projection) and smaller on the weakest distant agreement route (.576–.968). These cells share lexical examples, so60 is not a count of independent replications or a statistical significance claim. All per-item records remain available; no winning model/seed was selected.

Fixedgain2 can also improve effects, but may overshoot and does not reliably beat the parameter-free norm-restoration rule. The useful result is not that larger edits always work: scale must be judged against the true contrast, downstream KL and individual-effect errors. H855's task-coordinate constraint was unnecessary for much of this improvement.

**Important limitation.** This is a pairwise oracle, not a repaired single-input featurizer. It requires the full donor-recipient contrast and both active supports. In general, such pairwise corrections need not be expressible as F(donor)-F(recipient) for one consistent encoding function; they can violate cycle consistency. It also increases Euclidean reconstruction error compared with orthogonal projection. Thus improved behavior does not establish a data-canonical representation or an interpretable atom/group.

**Updated hypothesis tournament (20 alternatives).**

1. Reconstruction-optimal projection attenuates useful contrasts — strongly supported byH854–856.
2. The frozen sparse spans entirely lack task directions — weakened; constraints are feasible and norm restoration often works.
3. Full512-dimensional capacity is required — weakened for these task effects; active spans suffice after calibration in many cases.
4. Encoder coefficient error alone is the bottleneck — weakened; independent stateLS gives mixed causal results.
5. Learned support selection loses useful content — still plausible; full-span oracles outperform active spans, but objective/dose confound attribution.
6. Learnedshared membership collapse is a primary defect — supported as one defect, not the sole explanation.
7. Simply fixing coverage solves shared groups — weakened by fixedshared underperformance.
8. A task-aware metric is essential to improve fidelity — weakened for pairwise contrast calibration, unresolved for single-state learning.
9. A single scalar completely determines causal output — contradicted byH850 partial-writer counterexamples.
10. A compact task-specific axis summarizes many natural donor contrasts — supported byH848–849, within limited tasks.
11. One generic axis explains both agreement and sentiment — weakened by low overlap and cross-task controls.
12. The axes are distributed across learned atoms — compatible with weak individual atom alignment and useful aggregate spans.
13. Context features are written by multiple modules — supported byH850 native decomposition/interventions.
14. Early final-position state is the universal location — contradicted by depth/source-position results.
15. Agreement's relay depends on adjacency or wording — supported as a limitation; moving-relay versus signal-loss distinction open.
16. Sentiment's relay is strictly adjacent-token copying — weakened by retained effects with inserted words; broader formats remain untested.
17. Persistent lexical values plus contextual readers are reusable computation units — supported on prior panels, not the whole model.
18. Token-mean coverage explains all featurizer failures — weakened by same-token and correct-site results, but affects nonlinear encoding.
19. A per-state norm channel can reproduce the pairwise calibration benefit — live and directly testable next.
20. Pairwise behavior preservation may not define a consistent representation/manifold — unresolved; test single-state encodings and cycle consistency before claiming canonical form.

**Next experiment.** Try a direction-free, single-state inference rule: retain native residual norm as one additional scalar, rescale learned or least-squares reconstructions around their decoder bias, then form donor differences normally. Freeze supports and dictionaries, evaluate both natural-data reconstruction/operator fidelity and all existing task cells. This is an implementable encoding function, unlike the current pairwise oracle. Do not tune gain on these task outcomes. If it fails, investigate contrast-aware or response-weighted training with independent task confirmation rather than declaring the featurizer repaired.

**Audits.**4,992 full forwards; restored edit norms equal true contrast norms to float64 precision (max ratio1.0000000000000007); span residual<=2.25e-15; orthogonality<=4.45e-15; max active dictionary condition4.98. Identity logits exact; full reference margins replay within1.44e-5; sources/checkpoints unchanged; no reserved-test use; natural-document pool26,112 unchanged. H855–856 together used13,260 forwards and moved the explanation from a supposedly missing semantic coordinate toward attenuation plus remaining support/interaction limitations. Broad goal remains active.
