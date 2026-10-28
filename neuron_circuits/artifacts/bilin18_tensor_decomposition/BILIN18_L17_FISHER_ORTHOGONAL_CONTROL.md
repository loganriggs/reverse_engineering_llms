## H501 — The retained subspace and the rule for reading its coordinates both matter

Use exactly the H500 new donor sources and H499 receiver contexts, with all fitted maps frozen. For each head, the Fisher map has a four-dimensional write subspace spanned by B C^(1/2) U. Construct an orthonormal basis Q for this same subspace and replace the oblique map by ordinary Euclidean projection Q Q^T. Compare native finite responses under PCA, this orthogonal-write control, and the original oblique Fisher map. Both signs and all nine heads are retained. This is an exploratory mechanism diagnostic on reused outcomes, not another fresh confirmation.

The control retains the Fisher-selected write subspace and removes its particular coordinate-reading rule. Orthogonal projection minimizes raw Euclidean source error within that fixed subspace. The original oblique map reads C^(-1/2) U coordinates and writes through C^(1/2) U; a source component outside the write subspace can therefore change the reconstructed coordinates. This extra freedom can preserve behavior even when it worsens raw source reconstruction.

| Head | Sign | PCA KL recovery | Orthogonal Fisher-write recovery | Original oblique Fisher recovery |
|---|---|---:|---:|---:|
|0|negative|24.01%|18.72%|21.11%|
|0|positive|26.61%|25.40%|27.56%|
|1|negative|79.47%|78.61%|82.78%|
|1|positive|78.23%|77.41%|81.19%|
|2|negative|37.38%|35.80%|48.54%|
|2|positive|41.11%|41.39%|53.30%|
|3|negative|35.02%|53.76%|68.23%|
|3|positive|33.48%|50.14%|64.94%|
|4|negative|83.44%|83.02%|82.54%|
|4|positive|82.61%|82.41%|82.50%|
|5|negative|22.19%|42.50%|57.79%|
|5|positive|20.51%|41.14%|57.52%|
|6|negative|98.04%|98.45%|98.69%|
|6|positive|97.65%|98.17%|98.46%|
|7|negative|39.26%|43.55%|51.03%|
|7|positive|39.41%|44.35%|51.69%|
|8|negative|69.75%|74.40%|82.62%|
|8|positive|69.54%|74.05%|82.38%|

Head 5 illustrates both contributions: PCA recovers 20.51–22.19%, ordinary projection into the Fisher-selected write subspace recovers 41.14–42.50%, and the original oblique map recovers 57.52–57.79%. Head 3 similarly progresses from 33.48–35.02%, to 50.14–53.76%, to 64.94–68.23%. These point estimates support both better subspace selection and a useful coordinate-reading rule in these heads.

The pattern is heterogeneous. In heads 1 and 2 the orthogonal-write control is roughly comparable to or slightly worse than PCA, while the oblique map is better. In heads 7 and 8 both changes help at the point-estimate level. Head 6 changes little because all maps are already highly faithful. Head 0 remains poorly captured, and head 4 shows no consistent benefit. Oblique exceeds its orthogonal-write control in 17 of 18 point comparisons; head 4 negative is the exception. No new superiority intervals were specified for this diagnostic, so these counts are descriptive, not significance claims.

Raw source reconstruction always improves or stays equal when replacing an oblique projection by the orthogonal projection onto its own write span; the numerical audit checks this. Yet behavioral recovery often falls. Consequently, a useful computational description may require a read function together with a write direction, not just an axis and its Euclidean projection coefficient. This is compatible with many equivalent coordinate systems within the retained subspace. It does not establish that these particular four axes are uniquely canonical, semantically interpretable, or fundamental to all layers.

The experiment separates selected write subspace from coordinate readout, but not every reason the readout works. Candidate explanations include local output-Fisher weighting, compensation for omitted source components, source covariance correlations, context-dependent sensitivity, and finite normalization effects. The current maps use a local metric averaged over only 16 training receivers; finite native behavior may differ from that local objective. A direct local-versus-finite diagnostic is the next informative test.

Audits: QR bases are orthonormal and reconstruct the original write span within 1e-10; orthogonal source squared error is no greater than oblique error; native effect KL replays H500; archived rows are finite and replay aggregate control recovery. No additional documents were used; tracked total remains 19,456. Native downstream MLP computation remains a dependency.

Next H502: compute the exact local output-Fisher error for the PCA, orthogonal-write, and oblique source maps on the fixed H500 donor/H499 receiver pairs. Compare these local predictions with native finite intervention KL at prescribed scales 0.1, 1, and 2, for both signs and every head. Keep maps frozen. This separates local metric transfer from finite-intervention nonlinear effects and checks whether the readout advantage is already predicted near zero. Record native effect sizes and numerical small-effect limitations; this will be exploratory on reused data. The updated twenty-hypothesis tournament is saved separately. Fresh confirmation should target a specific discriminating claim after this diagnostic, not repeatedly tune to the existing panels.
