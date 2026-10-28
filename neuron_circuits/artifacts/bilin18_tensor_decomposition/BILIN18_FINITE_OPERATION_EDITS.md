## H657 — Finite edits improve output fidelity partially; KL and task loss diverge

H656 predicted local derivative changes for copying→task instructions. H657 performs actual forward passes on the96 held-out copying prompts, editing only the final-position normalized inputs of MLP10–17. Frozen original-wording same-task means, second-wording discovery means, other-task means, norm-matched other-task means and constant-answer means are added to each evolving MLP input. Inputs are not renormalized after editing. The rest of the model runs normally, including downstream attention and normalization.

Two donor-informed controls distinguish adding a native context difference from replacing the evolving context. Exact-delta adds the natural task context minus the natural copying context. Exact-clamp interpolates the evolving input toward the natural task context; at amplitude1 it overwrites that input exactly. Neither replaces earlier computation or all attention. Both access held-out donor states and are controls, not deployable predictors.

All seven edit types use the fixed amplitude grid0.25,0.5,1. No amplitude is selected or tuned on held-out results. The same four held-out items per family and four forms are retained. Native copying and natural task outputs provide baselines.

**Output fidelity:** KL(task||edited) is the full-vocabulary KL divergence from the natural task-prompt distribution to the edited copying distribution, in nats. Smaller is closer; it does not mean more correct. The target task prompt can itself be wrong.

| Family | Native copy KL | Same-task0.25 | Same-task0.5 | Same-task1 |
|---|---:|---:|---:|---:|
|addition|0.142|0.106|0.084|0.076|
|agreement|0.184|0.126|0.101|0.145|
|antonym|0.138|0.125|0.126|0.157|
|capital|0.204|0.161|0.123|0.127|
|plural|0.168|0.146|0.131|0.126|
|successor|0.506|0.380|0.330|0.461|

Original-wording same-task edits improve mean KL in every family at amplitudes0.25 and0.5. At amplitude1 they improve five families, but worsen antonyms. Stronger edits are not uniformly better, consistent with finite rollout effects and incompleteness of the shifted component. This does not establish a universal optimal strength.

**All methods at the prespecified full amplitude1:**

| Family | Original same-task | New-wording mean | Other tasks | Norm-matched other | Constant | Exact delta | Exact clamp |
|---|---:|---:|---:|---:|---:|---:|---:|
|addition|0.076|0.065|0.118|0.127|0.166|0.063|0.039|
|agreement|0.145|0.139|0.161|0.211|0.218|0.134|0.075|
|antonym|0.157|0.101|0.147|0.153|0.230|0.090|0.078|
|capital|0.127|0.085|0.166|0.187|0.304|0.089|0.089|
|plural|0.126|0.136|0.180|0.160|0.272|0.104|0.066|
|successor|0.461|0.460|0.473|0.638|0.640|0.550|0.159|

Full-amplitude same-task transfer beats pooled other tasks in five families, with antonyms again the exception. Controls and second-wording means have their own nonmonotonic responses. Exact-delta does not reliably solve the problem: successor full-amplitude KL0.550 is worse than native copying0.506. Exact-clamp does improve every family, but leaves nonzero divergence0.039–0.159. The exact context difference is therefore not a sufficient additive finite edit, and even exact later-MLP context replacement is insufficient to reproduce task-prompt outputs. Attention and earlier residual state remain plausible missing components.

**Task loss is a separate objective.** Target cross-entropy is minus log probability of the declared answer token, in nats. Lower can disagree with higher output fidelity.

| Family | Native copy CE | Natural task CE | Same-task edit CE at1 | Exact clamp CE at1 |
|---|---:|---:|---:|---:|
|addition|3.936|3.747|3.820|3.746|
|agreement|9.107|8.202|8.348|8.359|
|antonym|5.387|5.593|5.021|5.257|
|capital|6.132|4.986|6.091|5.448|
|plural|5.963|5.199|5.758|5.469|
|successor|3.911|2.207|3.749|2.962|

The clearest disagreement is antonyms: full same-task editing decreases target CE from5.387 to5.021 while increasing task-output KL from0.138 to0.157. The natural task distribution itself has worse target CE5.593 than native copying5.387. In plurals, the other-task edit achieves lower target CE5.409 than same-task5.758, while producing worse fidelity KL0.180 versus0.126. These are concrete examples of output-faithful and loss-oriented objectives ranking edits differently. No claim that either objective is universally preferable follows.

**Correctness:** every edited condition and both natural baselines have0% exact next-token correctness on this held-out subset. Candidate accuracy for native copy and natural task is addition0%, agreement50%, antonyms100%, capitals100%, plurals100%, successor0%. Most edits leave those rates unchanged; same-task amplitude1 changes agreement to56.25%. Candidate sets omit many incorrect outputs, so high candidate accuracy already present in copying does not demonstrate task execution. The broader panel's successor9.375% exact accuracy came from discovery items; it must not be transferred to this held-out subset.

These results move beyond derivative fidelity: some learned shifts produce useful finite changes in output distributions, but the evidence remains partial. The model retains its original weights, native earlier computation and full receiver context. This is neither a successful task transplant nor a canonical reduced model. It supports studying distribution-relative approximations with separate internal, output-KL, loss and correctness criteria, as proposed by the user.

Next localize the finite residual gap with an exact-factor intervention: donor final-position residual after block9, donor later attention outputs, and donor later MLP inputs, individually and jointly. A full combination provides a recovery check; individual comparisons reveal whether upstream state and attention are needed alongside MLP context. Use exact controls before fitting a more elaborate compact surrogate, and keep KL/CE/correctness separate. These are off-manifold intervention tests, not automatically causal shares of the natural task.

Audits:2208 completed whole-model forwards, comprising native replay, zero-strength hook replay, and21 edited conditions on96 prompts. A first preflight stopped after96 native forwards because batch-dependent floating-point replay deviation1.86e-5 exceeded the initial1e-5 threshold. It produced no intervention results. The rerun allowed1e-3 and observed the same1.86e-5 maximum deviation; zero-strength hooks exactly reproduce the rerun native outputs. Thus total work including the stopped preflight is2304 forwards. No numerical failure was treated as scientific evidence. Runtime source/mean/capture hashes unchanged and all scores finite. Fixed amplitude grid and all rows retained. FineWeb remains26,112 unique documents. Broad research goal active and incomplete.
