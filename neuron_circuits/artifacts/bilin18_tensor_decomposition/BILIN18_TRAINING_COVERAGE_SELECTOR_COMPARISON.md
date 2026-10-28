## H1042 — Expanded causal selector pilot improves weekday transfer but not general fidelity

Previous goal turn was progress: H1041 bounded reserved-format repair and froze the joint comparison. This turn executed six finite-causal fits: narrow five-family versus expanded ten-family training, crossed with scalar, disjoint-block, and shared-union selectors. All untrained controls also evaluated; no architecture selected.

Data: narrow60pairs/120unique prompts; expanded80pairs/160unique prompts, with the20H1039eligible additions. Exact cross-source training/evaluation prompt overlap0, duplicate endpoint aliases0, duplicate pairs0. PCA is refitted on training only using equal family weight, equal unique prompt weight within family, and uniform tokens within prompt. All three streams have rank128 and train-only center/scale. Unlike H1033's endpoint-pair weighting, this explicitly weights unique prompts; all current comparison arms use the same recipe. Baselines are newly fitted, not claimed numerically identical to H1033.

Architecture remains limited:384learned diagonal importance coefficients; fixed PCA coordinate values, fixed random partitions for groups. Scalar selects16coordinates perstream; block selects4groups of4; shared selects2groups of8 and decodes their union once. Maximum48active coordinates; shared actual union can be40–48. This is not a test of learned semantic memberships, a full learned BSF dictionary, or a computational DAG. Equal maximal sparsity is not exact equal union size.

Optimization: single seed10331, Adam0.03,80fixed pair steps, equal-family round robin, final checkpoint only. Actual finite downstream objective sums native-to-reconstruction KL and full-donor-to-decoded-difference-patch KL in both directions, plus1e-4importance penalty. Hard support forward with the existing straight-through surrogate. Native suffix frozen and excluded from inference. No convergence claim.

Important exposure limit:80steps gives16visits per family in narrow training versus8in expanded training. Narrow schedule covers all60pairs, expanded only56of80pairs directly under causal loss, although all80pairs enter training geometry. The comparison controls total compute, not exposure per family. It must not be presented as a complete test of fully trained broad supervision.

Paired relative KL, lower better, after finite-causal training:

| Evaluation panel | Scalar narrow→expanded | Block narrow→expanded | Shared narrow→expanded |
|---|---:|---:|---:|
| Same-family lexical |0.5265→0.7082|0.7750→0.8373|0.8863→0.8844|
| Historical copying/spatial |0.9125→0.9102|0.9433→0.9447|0.9675→0.9628|
| Reserved weekday development |0.8709→0.7743|0.9703→0.9190|0.9601→0.9821|

Metrics average families equally within panel; copying/spatial records pool their two competent historical cohorts within each family. Whole-output KL improves on historical exclusions (scalar4.2434→3.8656) and weekday (scalar5.2854→3.3206), but correct top-answer accuracy remains0for every method on both excluded panels. Same-family scalar whole-reconstruction macro accuracy falls32.5%→5.8%. This is partial distributional improvement, not successful model replacement or general reverse engineering.

Untrained controls distinguish geometry from optimization: scalar same-family paired error already worsens0.5422→0.7019when training geometry broadens, before any importance learning. Scalar weekday already improves0.9107→0.7988untrained; finite supervision changes it to0.8709→0.7743. Thus the scalar coverage tradeoff cannot be attributed solely to fewer causal updates. Block/shared controls show heterogeneous effects; no consistent shared-group advantage emerges. These small single-seed cohorts do not support architecture ranking in general.

Evaluation remains opened diagnostics:19same-family pairs,16historical excluded pairs,4adaptively screened weekday pairs,78endpoints. No evaluation prompt enters PCA or optimization. No fresh-task validation or unlabeled discovery claim. Coarse boundary-port native assistance remains present only through the frozen suffix and intervention baseline; candidates here must represent all three streams, unlike the earlier residual-only diagnostic.

Verification:3952suffix evaluations including1920differentiable student passes, six completed fits, finite losses/gradients/metrics, frozen source hashes unchanged. PCA orthogonality3.56e-15, max48active. Independently recomputed all summaries and KL from36saved logit examples (one endpoint per panel/arm), max discrepancy4.36e-14. All candidate checkpoints and training geometry saved.

Next bounded control: continue each expanded selector to160total fixed steps with the same schedule/order and no hyperparameter changes. This provides16visits per family, matching narrow80 exposure and covering all80expanded pairs, at twice the optimization compute. Compare against both saved expanded80 and narrow80; report the compute difference. Do not select a stopping point using evaluation. If original-family degradation persists, the untrained controls already suggest rank/sparsity allocation or task-conditioned geometry rather than merely undertraining. Avoid declaring broad supervision ineffective from this short compute-matched pilot, and avoid another unconstrained rank sweep.
