## H696 — Perfect hierarchical reconstruction does not recover causal refinements

Train a seven-node hierarchy: root, two intermediate categories, and four leaves. Each node contributes two independent teacher directions, all14directions orthonormal in20dimensions. Inputs contain the refinements of every ancestor implied by their seed nodes. Train on all singleton seeds and all leaf pairs; hold out leaf triples and all four leaves. Fit nine models:three seeds×taxonomy union, additive refinements and flat expanded-gate control. Every model has594trainable parameters, learns its encoder and decoder, and receives the hierarchy/gates as side information. Same initial weights and minibatch streams,3000Adam steps. No teacher coordinate targets used in fitting.

Taxonomy union has nested memberships containing all ancestor directions and decodes their union once. Additive refinement has disjoint per-node coordinates and sums ancestor-plus-child terms. The flat control receives already-expanded ancestor gates. These are deliberately functionally equivalent for this supplied support structure; they are not competing discovery algorithms. Their equivalence prevents claiming hierarchical superiority merely from reconstruction. Initial outputs agree within2.38e-7; trained results are also nearly identical.

| Seed | Taxonomy held relative MSE | Mean non-root subspace overlap | Root removal relative effect error | Mean non-root removal effect error |
|---|---:|---:|---:|---:|
|31|1.58e-13|0.715|2.547|0.574|
|32|6.07e-12|0.708|1.824|0.698|
|33|2.14e-13|0.719|2.374|0.770|

All nine fits reconstruct held combinations to relative MSE below7e-12. Yet non-root decoder subspaces have teacher overlap only0.486–0.875. Overlap is the average squared cosine of the two principal angles, invariant to rotations within each block.

Interventions remove a node's encoded reconstruction contribution while holding all other codes/gates fixed. The reference effect is that node's true teacher contribution for the same input. Non-root effect errors are substantial: squared error divided by teacher-effect energy ranges0.148–1.584. Between12.9%and52.8%of non-root removed energy lies outside the intended teacher refinement subspace. These are synthetic ground-truth checks, not LLM intervention claims.

The root is especially informative: its decoder subspace matches the true root almost perfectly, with negligible removed energy outside that subspace, yet removal-effect error is1.82–2.55. Correct decoder geometry does not guarantee correct coordinates or contribution attribution. Other blocks can contain ancestor directions that cancel root-coordinate terms in the total reconstruction. Observed reconstruction and intervention errors support this compensation explanation; a constructive invariance audit is the next way to establish the precise freedom.

This is why the user's requested manifold verification must include more than reconstruction or stable subspaces. Hierarchical activation constraints encode implications, but do not by themselves force child coordinates to mean ancestor-independent refinements. Additional constraints or interventions may select such a decomposition, and must be justified rather than assumed. Orthogonalizing child refinements could match this orthogonal teacher while imposing the wrong assumption on a real model.

Audits: source hashes unchanged, finite results, all variants matched594trainable parameters, initial forward equivalence checked. Existing ten feature-structure correctness tests passed again. No original-model forwards/new documents; reserved LLM test split untouched. Known hierarchy/gates remain supplied; topology and joint gate discovery are still outstanding.

Next construct an explicit ancestor/child mixing transformation that preserves observed hierarchical reconstructions but changes component-removal effects. Compare an ancestor-residualized convention on orthogonal and deliberately nonorthogonal teachers before deciding whether it is useful as a canonicalization rather than discovery of unique true units. Continue computational-DAG reuse tests and return validated representations to early-layer geometry/causal analyses. Broad goal active.
