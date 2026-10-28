## H1045 — Direct causal gate training worsens training loss as well as transfer

Previous goal turn was progress: H1044 established a direct-change encoding advantage and proposed a matching training objective. This turn fitted three direct-change selectors from zero importance, using expanded80pairs/tenfamilies,160fixed steps, all pairs covered, unchanged geometry/group partitions, and final checkpoints only.

Objective is the average finite donor-output KL after direct encoded changes, in both directions, plus1e-4importance regularization. Whole-state reconstruction loss is removed. Adam0.03, clamped log-importance[-2,2], hard top-k/union forward and the existing softmax straight-through surrogate. Models learn384importance weights; they do not learn dictionary directions or memberships. Evaluation receives both endpoints and native recipient context. No evaluation examples used for fitting.

Paired relative KL, initial untrained→final direct-causal:

| Panel | Scalar | Block | Shared |
|---|---:|---:|---:|
| Same-family |0.4475→0.4575|0.6322→0.6403|0.7161→0.7909|
| Historical copying/spatial |0.8448→0.8556|0.9166→0.9205|0.9220→0.9391|
| Weekday development |0.7498→0.7744|0.8416→0.8532|0.8974→0.9173|

All nine comparisons worsen. Same-family donor-correct top1declines22.5%→20%scalar and10%→5%block; shared remains0%. Excluded copying/spatial stays0%for all, weekday12.5%. These are paired-edit accuracies, not whole-state replacement accuracy.

Crucial follow-up: evaluated exact hard-forward loss on every training pair at initialization and final weights, with equal family weighting. This was a read-only training-fit audit, not extra optimization or checkpoint selection:

| Selector | Initial training KL | Final training KL |
|---|---:|---:|
| Scalar |0.3281|0.3630|
| Block |0.8283|0.8692|
| Shared |1.0946|1.2154|

Thus the problem is not merely failure to generalize a successfully fitted training objective. This recipe does not even lower its full training objective after160steps. The straight-through gradient is a surrogate, not the derivative of the discrete hard-forward objective. Biased gradient directions, finite-step instability, and cyclic-family interference remain plausible. This result does not prove a coding error or isolate those alternatives. It also cannot be used as evidence that causal supervision itself is ineffective or that the dictionary is the sole bottleneck. Changing geometry without first understanding this fitting failure would confound representation and optimizer conclusions.

The direct-encoding improvement of H1044 remains valid and came without learning. Frozen untrained sparse encoding is the stronger control here; do not promote these trained weights. No robust architecture ordering, general manifold validation, or new-task discovery is established.

Verification: untrained direct controls reproduce all H1044paired KL exactly. Frozen source/geometry hashes unchanged; three fits, all80pairs, max48/min43active coordinates. Checked direct zero-change and reverse-sign consistency exactly.1588suffix calls for fits/evaluation, including960differentiable student calls; separate full-training objective audit1120calls,2708total. Independently recomputed KL from18saved evaluation logits, max4.27e-14. All losses/gradients/metrics finite; no train/evaluation prompt overlap. Optimizer states saved.

Stop this diagonal-importance training recipe rather than extend steps or search evaluation thresholds. Next is a bounded optimization-mechanism audit on training data only: compare surrogate-predicted descent with actual hard-forward loss changes around initialization/current weights, check soft-surrogate gradients independently, and separate support changes from coefficient/decoder correctness. Use a fixed diagnostic batch and preserve full-training audit evidence. No new representation winner can be claimed from these unsuccessful fits. A future learned dictionary/group experiment needs an optimization control that demonstrably lowers its actual training objective before interpreting transfer or manifold quality.
