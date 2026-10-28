## H836 — Balanced overlap repairs coverage but does not close the prediction gap

H835 exposed an invalid inference risk:512 membership slots for512 atoms meant overlap necessarily reduced coverage. H836 allocates1024 slots (128 groups x8 memberships) and constructs two independently randomized partitions, so every atom initially belongs to exactly two groups. Compare fixed balanced memberships with learned memberships from that initialization. Keep512 atoms and Top8 active groups. Add signed scalarTop48 and Top64 controls, retain block8 Top8, and extend training to2400 steps.

### Design and implementation checks

Twenty fits: two pre-MLP sites, five model types, two seeds8351/8352. Reuse seeds and minibatch streams so scalar64/block8 extend their prior baseline trajectories. Same token-residual training data, lexical baseline, scale access, Adam schedule and development checkpoint criterion. No H834 confirmation examples enter model selection. Fixed memberships are supplied as a buffer and excluded from learning; learned memberships use straight-through Top8-per-group selection.

Structural tests verify initial row degree8, column degree2, full512-atom coverage, exact union decoding, correct Top48/Top64 supports and gradient flow to coordinate/group encoders and learned membership logits. Frozen membership logits receive no gradient. Final trained union discrepancy is exactlyzero and decode formula discrepancy2.38e-7. All input/model/implementation hashes remain unchanged. All20 fits and per-document response predictions are saved; no full-model/test forwards or new natural documents, pool26,112.

### Development response prediction

Relative RMSE on H832's512 development documents and fixed evaluation probes:

| Site | Scalar48 | Scalar64 | Block8 (64 active) | Fixed balanced shared | Learned shared |
|---|---|---|---|---|---|
| block1 |.577,.579|.556,.554|.613,.610|.708,.707|.715,.711|
| block2 |.609,.608|.583,.584|.648,.650|.763,.762|.772,.770|

Both scalar budgets outperform block8 and the two shared variants in this finite-training comparison. Increasing training from800 to2400 steps improves the scalar/block baselines substantially. This is a development result, not independent confirmation or proof of intrinsic architecture dominance. Scalar models are signed TopK dictionaries, not conventional ReLU SAEs; blocks do not impose Grassmannian orthogonality.

Fixed random overlapping groups are an expressivity/optimization control, not a claim that their initial grouping captures meaningful concepts. Their poor score cannot rule out learned, well-optimized overlapping group structures. It does show that full coverage and near-matched active dimensions alone do not eliminate the gap in the implemented setup.

### Coverage and actual budget audit

Fixed shared models preserve all512 atoms in the membership graph, use all512 atoms and all128 groups somewhere on validation, and average61.3–61.7 active atoms per example. They therefore repair the specific H835 coverage failure. The gap remains despite active counts close to64.

Learned memberships again concentrate: selected checkpoints cover102/98 atoms at block1 and85/89 at block2, with average43.4–45.5 active atoms. All128 groups are used on validation, so this is not merely unused gates; memberships increasingly reuse a small atom subset. All covered atoms are used at least once. There is enough total membership capacity for full coverage, so the remaining loss is a learned-membership optimization/regularization issue rather than the earlier unavoidable slot-count constraint.

Scalar48 is a closer, but not exact, active-count comparison to learned shared models. It still predicts better, so the earlier64-versus48 budget difference is not sufficient to explain the observed gap. Parameter counts remain unmatched: scalar/block1,181,312; shared1,394,432 registered parameters, of which1,328,896 are trainable for fixed memberships and all1,394,432 for learned memberships. The fixed implementation additionally stores its balanced membership buffer; inactive membership logits remain serialized. Do not equate registered parameter counts with deployment storage or runtime. Shared gate inference adds computation.

### Training status

Scalar/block runs select2400 steps and generally still improve, although recent gains are smaller. Fixed shared runs select2400 at block1 and2200 at block2; learned shared select1800–2400 depending on run. Some shared curves plateau or fluctuate. This is evidence of different optimization behavior, not a convergence proof or a reason to select only the best seed. All curves and selected checkpoints remain available.

### What changes in the research direction

The shared decoder's set semantics are verified; learned membership discovery remains weak in these pilots. Adding overlap and gate capacity does not automatically reveal a better computational decomposition. The strongest current local predictor is the signed scalar model, while larger blocks could still have more interpretable or stable geometry despite lower prediction fidelity. Those are separate claims requiring separate tests.

Freeze these trained candidates now and evaluate new document-disjoint occurrences, including seen/unseen-token strata and fresh probe directions. Carry a fixed subset through full-model output KL with own-response replay and zero-response controls, so local derivative improvements cannot substitute for downstream faithfulness. Retain block and fixed-shared models as competing representations in manifold/causal-transfer tests rather than discarding them solely on reconstruction ranking. Learned membership collapse should be addressed by explicit degree/coverage constraints or data-informed memberships in a future controlled branch, not hidden by changing budgets during confirmation. The user's broader diverse-task, selective-reuse and computational-meaning requirements remain outstanding.
