## H755 — Scalar sparsity beats matched block TopK reconstruction in this pilot

Twelve task-blind fits compare signed scalar TopK (eight selected coordinates) with signed block TopK (two four-dimensional blocks), for L5H3/H5 and three seeds. Both have128 encoder/decoder coordinates,33,024 parameters, identical initialization seeds, identical minibatch sequences, global train-RMS scaling in H754 write coordinates, and3,000 Adam steps at0.001 with batch512. Checkpoints are selected by natural validation MSE every250steps. No task examples enter fitting.

| Head | Scalar8 validation write MSE, three seeds | Block4×2 validation write MSE, three seeds | Adaptive PCA8 reference |
|---|---|---|---:|
| L5H3 | 41.31%,41.15%,41.02% | 55.41%,55.23%,55.08% | 51.18% |
| L5H5 | 39.15%,38.82%,38.76% | 50.70%,50.42%,50.73% | 43.20% |

Scalar TopK improves over adaptive PCA; fixed blocks perform worse than both on reconstruction. Equal active dimension and parameter count do not equalize combinatorial support flexibility. This block implementation uses unit decoder rows and block-norm gating; it is not Goodfire's orthonormal-within-block Grassmannian implementation. The result does not reject all block/manifold models, shared-atom groups, or hierarchical alternatives.

Every run selects step3,000 and still improves over step2,750, by0.04–0.28 percentage points of relative MSE. Therefore these are finite-training results, not established optima. There are no dead validation blocks in any run.

Seed consistency is limited. Mean matched scalar decoder absolute cosine is0.657(H3)/0.558(H5); block subspace mean squared principal cosine is0.325/0.297. Those are different metrics and should not be directly ranked against each other. Matched activation-support Jaccard is0.283/0.155 for scalar atoms and0.164/0.104 for blocks. Reconstruction gains do not imply uniquely identified computational units.

Audits: identical33,024 parameter counts; every validation sample uses exactly eight selected scalar slots; explicit decoder formula exactly matches forward output; unit decoder norms verified; finite losses and gradients throughout; data/implementation hashes unchanged. Zero whole-model or reserved-test evaluations. Natural document pool remains26,112. Per-fit checkpoints, histories and all seeds are saved.

Next: compare full-head operation swaps with reconstructed differences F(donor)-F(receiver), measuring both total behavioral change and the dependence specifically mediated by the verified downstream component. Include adaptive/fixed PCA and random controls. This causal comparison, not reconstruction ranking, determines which representations merit manifold and shared-group tests.
