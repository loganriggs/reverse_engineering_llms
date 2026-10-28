## H722 — Task-blind head compression separates functional preservation from reconstruction

Previous turn classification: progress. H721 identified strong native-head causal baselines, agreementL11H3andcapitalL13H0. H722 returns to representation fitting inside those verified128Dcomponents.

Capture their pre-output-projection head coordinates on the same natural-text corpus split used earlier:2048training documents×16positions=32768vectors per head, then256validation documents×16positions=4096vectors. Do not touch the original256document reserved test. Fit centered float64PCA on training vectors only; compare one seeded random orthogonal basis per head with nested ranks1,2,4,8,16,32,64,96,128. No task labels or task activation vectors enter the fit. The choice of heads remains task-informed, and validation/task panels are reused research panels.

Two different functional tests are kept separate. On every frozen task/nuisance pair, project the donor-minus-receiver head-coordinate difference into the retained subspace, add it to the receiver's native coordinates, and recompute downstream normally. This tests preservation of a causal difference, not replacement of every native activation. Separately, on32hash-selected validation documents atposition63, replace the entire head activation by training_mean+projection(native-training_mean), and measure KL(native output||reconstructed output) and observed next-token CE change. General fidelity is evaluated at ranks8,32,64,128. Fullrankmustreplay the native head and exact task swap.

| Head | PCA rank | Validation coordinate variance retained | Fraction of full-head transfer task gain | Rank-matched random task fraction | Mean natural-text output KL after head reconstruction |
|---|---:|---:|---:|---:|---:|
|agreement|8|24.4%|79.1%|1.6%|0.000993|
|agreement|32|53.7%|94.9%|16.8%|0.000428|
|agreement|64|75.7%|97.1%|57.1%|0.000167|
|agreement|96|90.2%|99.2%|84.6%|not evaluated|
|capital_recall|8|21.6%|10.4%|3.4%|0.007315|
|capital_recall|32|52.6%|46.7%|19.6%|0.002393|
|capital_recall|64|77.1%|74.3%|49.3%|0.000673|
|capital_recall|96|91.7%|94.5%|75.7%|not evaluated|

![Task effect versus activation reconstruction](/home/loganriggs/Documents/Codex/2026-10-21/he/outputs/verified_head_pca.png)

Agreement's task effect is concentrated in a relatively small natural-text PCA span. Rank4already retains68.9%of the full-head transfer margin effect; rank16retains89.1%,rank32retains94.9%. Yet rank32retains only53.7%of validation coordinate variance. Its transfer specificity is4.557versus4.761for the full head. A basis can therefore preserve this causal effect while leaving substantial activation variance unreconstructed.

Capital recall is different under the same procedure: rank32retains46.7%of the full-head task effect alongside52.6%of coordinate variance; rank64retains74.3%andrank96retains94.5%. Rank32specificity is1.527versus3.015for the full head. These observations support different degrees of functional concentration in these two components; they do not prove that capital recall intrinsically requires96dimensions or that agreement has a32Dmanifold. PCA is a fixed linear basis selected for coordinate variance, and another metric or adaptive representation may do better.

The projected residual write matters too. At rank32, agreement's write-space relative MSE is0.2743versus coordinate MSE0.4631; capital's corresponding values are0.4730and0.4741. Thus the learned output projection already suppresses some coordinate-reconstruction errors differently across heads. This motivates separating head-coordinate geometry from the geometry of the write delivered to the model.

Natural-output fidelity broadly improves with rank and beats the single random-basis baseline. At rank32, meanKLis0.000428agreement versus0.001183random, and0.002393capital versus0.007020random. These are averages over32documents and one head at a time, not guarantees of global model faithfulness. Observed continuationCE changes are retained separately in the JSON; neither CE nor this KL is task correctness. Task-effect fractions are relative to each head's own partial effect, not the model's full native competence. In particular retaining95%of a head effect does not mean95%accuracy.

Neither head is globally low-rank under the raw coordinate metric: training90%variance requires95dimensions foragreement and89forcapitals. Full whitening scale conditions are4.83and7.10, numerically manageable. This makes a full-span sparse-rotation/shared-group comparison feasible without inheriting the earlier256Dtruncation limitation.

Next establish an output-write-metric PCA baseline (fit on the head's projected residual write, map reconstructions back into head coordinates), then compare whitened sparse rotations/shared-group candidates at matched active dimensions. Evaluate frozen task/nuisance-effect preservation and natural-output fidelity alongside reconstruction; report all seeds and stability before treating groups as manifolds. This directly tests whether the current difference reflects the choice of metric, genuine need for adaptive supports, or a poorly matched representation family. It is not a reason to label PCAcomponents as semantic units.

Audits:2304fourteen-block prefix evaluations for natural-text capture,2272whole-model evaluations for task and native-output tests. Fullrank task swaps replay H721within3.82e-6answer-margin error; fullrank native reconstruction maxKL3.65e-12. Independent CPUfloat64rank32projection loss replay exact, basis orthogonality error2.11e-15. All source/model/data hashes unchanged and metrics finite. Original reservedtest untouched, no new unique research documents, corpus pool26112. Broad goal active.
