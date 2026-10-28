## H687 — Alternative feature structures implemented; correctness and first synthetic learning tests

User proposal implemented in scripts/feature_structures/models.py:

1. Shared dictionary B, separately encoded atom coordinates and sparse group gates, binary group memberships, exact union decoding so shared atoms contribute once. Column normalization is in the forward parameterization. Membership/group selection supports threshold or fixed TopK budgets with explicitly documented straight-through surrogate gradients. The objective exposes reconstruction and group/coordinate/membership costs. Nonzero coordinate costs count actual nonzero codes in the forward pass. Fixed budgets constrain costs rather than reproducing a learned variable-size penalty solution.
2. Acyclic child-implies-parent closure with multiple parents. Closure changes gates, not extra decoder copies. Shared-union decoding still counts atoms once.
3. Separate additive-refinement module where ancestor and child terms both contribute. This is a distinct hypothesis; architecture alone does not guarantee residual semantic refinement.
4. Ordered computational DAG with learned node/edge gates, explicit products of parent coordinates, one evaluation per node, shared consumers, and node/edge interventions. The topology is supplied; automatic graph discovery is not yet implemented.

Ten semantic/numerical tests pass: shared-atom union and exact counts (including zero coordinates), selective shared-atom removal, multi-parent ancestor closure without duplicate decoding, closure idempotence, invalid/cyclic edge rejection, additive refinement, finite nonzero surrogate gradients and column normalization, fixed-support finite-difference coordinate gradient, exact oracle reconstruction of unseen combinations, and nonlinear shared computation with selective node/edge ablation. Tests are in tests/test_feature_structures.py. Straight-through gradients are tested for connectivity/finiteness, not incorrectly compared with derivatives of hard discrete selection.

### Initial overlapping-group learning experiment

Synthetic activations use8 orthonormal atoms in16 dimensions, four groups of3 with overlaps. Training includes singleton groups and adjacent pairs; other pairs, triples and all-group activation are held out. True group gates are supplied to every architecture. Three seeds,3000Adam steps each. This isolates conditional representation learning; it is NOT unsupervised group recovery and is NOT a parameter-matched superiority comparison. Separate blocks have their own encoders/decoder coordinates; actual parameters and active coordinate counts are retained.

| Variant | Mean train relative MSE | Mean held-out-combination relative MSE |
|---|---:|---:|
|shared_known_membership|0.000000|0.034683|
|shared_learned_membership|0.073624|0.116531|
|separate_blocks|0.048296|0.193654|

Shared known-membership models fit training nearly exactly but retain2–5% held-combination relative error. Learned membership models retain about11–12%; separate blocks18–21%. These are preliminary conditional-learning results, not proof of recovered ground-truth atoms or advantage at matched budgets. Exact oracle dictionary/encoder parameters reconstruct unseen combinations to numerical precision in the correctness test. The trained known-membership failure therefore exposes an optimization/identifiability/generalization problem rather than inability of union decoding to express the target.

### Initial shared-computation learning experiment

A fixed five-node DAG is trained from random weights on outputs(2*x0*x1,3*x0*x1,x0), excluding the positive-positive input quadrant from training. The hidden product can be computed at one shared node and reused by two consumers. True node gates are supplied; topology is fixed. All three seeds generalize to the held-out quadrant with relative MSE below1e-8. Removing the shared product node gives large errors in both dependent outputs while leaving the independent x0 output near numerical precision. This validates trainable nonlinear reuse under known topology; it does not demonstrate unsupervised discovery of a DAG in LLM activations.

Hierarchy currently has semantic correctness tests, not yet a trained nested-structure recovery benchmark. Outstanding user requirements include genuinely unsupervised gate/membership recovery, learned or selected graph structure, matched reconstruction/active-dimension/parameter comparisons, selective reuse tests for learned shared atoms, and application to early LLM activations. These remain active work, not completed claims.

Next diagnose the held-combination failure with teacher-aligned recovery metrics, explicit permutation/sign alignment, synthetic oracle controls, and a budget grid. Add nested synthetic recovery and unsupervised gates before selecting alternatives for the LLM atlas. Retain the distinction between reusable support, taxonomic implication and reusable nonlinear computation throughout.

No original-model forwards or new FineWeb documents in this benchmark. All outcomes retained, including known-membership transfer failures. Implementation/tests are snapshotted after validation; synthetic runs use reconstruction-only training, so the subsequent exact-L0 reporting correction does not alter their trained models. Broad goal and the user's alternative-architecture request remain incomplete.
