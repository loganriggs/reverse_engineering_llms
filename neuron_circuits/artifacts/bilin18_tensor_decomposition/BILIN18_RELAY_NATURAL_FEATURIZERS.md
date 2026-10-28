## H852 — Task-blind layer4 fits favor scalar reconstruction; learned shared memberships collapse coverage again

All ten frozen representations trained from scratch on H851 natural-text token-centered normalized preMLP4 states. No relay task data or supervised directions entered fitting or checkpoint selection. Same architecture family, seeds, training step budget and initialization policy as H836; these are new correct-site models, not reused early-site decoders.

Signed scalar Top48/64 and block8 Top8 use512 atoms. Fixed/learned shared unions use512 atoms,128 groups,8members/group andTop8 group gates; two initial partitions give every atom membershipdegree2. The learned model uses the existing straight-through membership estimator. Shared atoms are decoded once through exact union support. Dictionaries have unit-norm columns/rows. Block8 is not a Grassmannian-orthogonal BSF, and signed scalar is not a ReLU SAE.

Training:2,400 Adam steps, minibatch128, learningrate .001; select the lowest natural-development residualMSE every200 steps. Both seeds retained. Most baselines select the final step and may still improve, so this is a finite-budget comparison, not proof of convergence or universal architecture superiority. Shared models have extra gate/membership parameters and actual active support differs.

|Model|Seed|Selected step|Development residual relative MSE|Mean active atoms|Atoms covered by groups/dictionary|
|---|---:|---:|---:|---:|---:|
|scalar48|8351|2400|0.4679|48.0|512|
|scalar48|8352|2400|0.4704|48.0|512|
|scalar64|8351|2400|0.4371|64.0|512|
|scalar64|8352|2400|0.4380|64.0|512|
|block8|8351|2400|0.5291|64.0|512|
|block8|8352|2400|0.5369|64.0|512|
|shared_fixed|8351|2400|0.7064|61.4|512|
|shared_fixed|8352|2400|0.7076|61.4|512|
|shared_learned|8351|2200|0.7408|47.2|101|
|shared_learned|8352|2000|0.7400|47.0|101|

For comparison, training-residual PCA64 has development relative MSE .6173 at the same centering. Scalars and block8 improve on it; both shared models are worse under this training setup. Fixed shared membership preserves all512 atoms, but learned memberships cover only101 and activate about47 atoms on average. This repeats the earlier coverage-collapse failure mode despite balanced initialization. A richer group architecture has not automatically produced a better representation.

Independent Gaussian-direction local MLP-response prediction has relativeRMSE .629 forscalar64, .653–.655 scalar48, .703–.707 block8, .808–.809 fixedshared, .834 learnedshared. These development diagnostics are not task-effect fidelity. They retain actual receiver radius and the original MLP weights, as inH836; no claim of a complete compressed model.

Audits: all ten fits completed; explicit decoder formula discrepancy<=2.39e-7; shared-atom union exact; source/data/implementation hashes unchanged; no whole-model or reserved-test forwards in fitting/evaluation. Task-independent local derivative evaluations do not fit task labels. Natural-document count unchanged. Next evaluate every frozen fit plus fixedPCA on H849 relay contrasts, separately reporting decoder alignment, actual encoded difference fidelity, and downstream causal-effect fidelity. Do not declare a semantic feature merely because a decoder subspace contains the supervised direction.
