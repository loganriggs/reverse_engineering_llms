## H863 — Does a component union require multiple computational directions?

H862 completed causal tests of nested feature unions, changing the next question: many dictionary atoms may be required because the dictionary is misaligned with a simple task signal, not because the model's computation is intrinsically high-dimensional. H863 splits every frozen ranked union into a task-aligned rank1 direction and its orthogonal complement, and evaluates both causally. The prior goal turn therefore made concrete progress; the broad research goal remains active.

### Exact experiment

All60 ranked unions from H862 are retained: five architectures, two seeds, two discovery-task axes, and nominal8/32/64 atom-slot budgets. Let Q be an orthonormal basis for a union and a the discovery task axis. Define v=QQᵀa/||QQᵀa||. For each native donor–recipient contrast δ, the full edit is QQᵀδ, the rank1 edit is v(vᵀδ), and the complement is their difference. These edits sum exactly in activation space, but their downstream effects need not add.

Patch each edit into the recipient's raw preMLP4 state, retaining the rest of the state and model computation. Test both agreement and sentiment, all156 recipient contexts from78 aligned H849 pairs. Also rescale the rank1 edit to the full union edit norm as a pairwise oracle amplitude control. No new basis fitting, encoder training, seed selection, or downstream outcome selection occurs. These directions remain discovery-task-supervised; this is not unsupervised feature recovery.

### Results at nominal64 slots

The reference in this table is the full union effect, not the full native relay swap. Rank1 slope1 means the rank1 effect matches the union's overall scale; RMSE measures per-example mismatch relative to the norm of union effects. Complement slope measures its isolated effect. Interaction RMSE measures union minus rank1 minus complement effects, normalized by union effect norm. All formats, distances and recipient orientations are included.

|Architecture|Seed|Task|Rank1 / union slope|Rank1 relative RMSE|Complement / union slope|Interaction relative RMSE|
|---|---:|---|---:|---:|---:|---:|
|block8|8351|agreement|0.987|0.064|-0.003|0.052|
|block8|8352|agreement|0.935|0.105|-0.005|0.105|
|block8|8351|sentiment|0.937|0.110|0.025|0.104|
|block8|8352|sentiment|0.927|0.120|0.028|0.101|
|scalar48|8351|agreement|0.957|0.085|-0.003|0.083|
|scalar48|8352|agreement|0.941|0.102|-0.005|0.098|
|scalar48|8351|sentiment|0.924|0.120|0.036|0.091|
|scalar48|8352|sentiment|0.960|0.100|0.018|0.117|
|scalar64|8351|agreement|0.973|0.080|-0.007|0.082|
|scalar64|8352|agreement|0.946|0.093|-0.005|0.097|
|scalar64|8351|sentiment|0.940|0.110|0.025|0.095|
|scalar64|8352|sentiment|0.932|0.146|0.022|0.111|
|shared_fixed|8351|agreement|0.928|0.098|0.000|0.096|
|shared_fixed|8352|agreement|0.949|0.084|-0.007|0.085|
|shared_fixed|8351|sentiment|0.946|0.114|0.028|0.098|
|shared_fixed|8352|sentiment|0.953|0.092|0.026|0.080|
|shared_learned|8351|agreement|0.963|0.108|0.002|0.076|
|shared_learned|8352|agreement|0.937|0.141|-0.002|0.104|
|shared_learned|8351|sentiment|0.948|0.111|0.009|0.087|
|shared_learned|8352|sentiment|0.851|0.294|0.085|0.105|

The saved analysis also contains all8/32-slot results, off-task results, per-gap strata, rank1 amplitude controls, output-distributionKL, and lexical-cluster bootstrap intervals. Off-task union effects can be tiny, so ratios there must not be read as stable effect sizes.

### Main finding and research decision

At nominal64 slots, the within-union rank1 direction reproduces92.4–98.7% of the union effect scale in19/20 own-task fit comparisons. The remaining learned-shared seed2 sentiment case reaches85.1%, with relative effectRMSE0.294 and a wide lexical-pair interval. The other19 relativeRMSEs range0.064–0.146. Agreement complements have isolated slopes between-.0073 and.0017; sentiment complements range.0089–.0849. InteractionRMSE remains0.052–0.117, so neither a perfectly linear sum of causal effects nor exact one-dimensional sufficiency is established.

Mean KL(union||rank1) ranges0.000128–0.000936 nats at64 slots. These distributions are close on this panel, but the comparison is to the union intervention, not to the unmodified model or full donor swap. The rank1 edit already retains about85–92% of the union edit norm on average, consistent with concentrated task contrasts. Rescaling it to full union norm worsens answer-effectRMSE in19/20 comparisons; the unscaled coordinate is generally more faithful than indiscriminate magnitude restoration here. This does not contradict H859's average benefits for full natural reconstruction: the operations, references and data differ.

The most likely interpretation is that these unions represent a simple task signal in an inconvenient dictionary basis. The result weakens the claim that needing many learned atoms is evidence for an intrinsically multidimensional computational manifold. It remains compatible with off-axis contextual modulation and with richer computation on other tasks. Because the axis and union were selected with discovery-task supervision, the experiment is a diagnostic upper bound on representation organization, not an unsupervised solution. Conditional concentration was plausible from discovery geometry; the additional evidence here is that it survives causal intervention and distributional comparison on the reused lexical/format/gap panel.

### Next step: necessity, then other computation sites

Stop treating further compression as the end goal. Freeze these rank1 directions and test whether removing their donor–recipient coordinate difference diminishes the native behavioral distinction. Use pair-midpoint coordinate erasure, full-union erasure and complementary-coordinate erasure, retaining native state elsewhere and evaluating both recipients. This tests local necessity of the contrast coordinate, rather than only its ability to transfer a signal. It must not be described as global model necessity because redundant routes and other sites remain intact.

Then extend the same behavioral verification to the already identified binding/copy/capital value-bus sites. Agreement and sentiment at one residual relay are insufficient to choose a general feature ontology. The user's broader manifold, reusable-computation, held-out-combination and diverse-task requirements remain open.


### What this test can and cannot establish

If a rank1 direction approximates a union, the union's useful task signal can be concentrated in a rotated coordinate. That weakens the inference that a large learned group collection establishes an intrinsically multidimensional computational manifold. It does not show the full original relay state is one-dimensional, that arbitrary language behavior is captured, or that the learned encoder discovers the useful coordinate without supervision. H862's full unions already retained only part of the full native relay effect.

Nonzero complement effects or interactions leave room for contextual modulation and additional computational directions. Even close answer-effect agreement is not identical to whole-distribution fidelity; KL(union||rank1) is saved explicitly. Rank1 rescaling uses donor-pair information and changes the edit, so any improvement from it cannot be attributed to an independent single-input featurizer.

### Verification and scope

37,752 full-model forwards. Native identity logits match exactly. Full-donor replay error1.43e-05; priorH862 full-union effect replay error0. Basis orthogonality error2.66e-15, decomposition error0, complement-axis dot error1.22e-11, and norm-control error2.73e-12. Source hashes unchanged and outputs finite. No reserved-test evaluations;26,112-document pool unchanged. Bootstrap intervals use4096 resamples of lexical pairs, only8 agreement and5 sentiment clusters, preserving repeated templates/gaps/orientations. Intervals are pointwise and uncorrected. These are reused research tasks, not fresh broad generalization evidence.
