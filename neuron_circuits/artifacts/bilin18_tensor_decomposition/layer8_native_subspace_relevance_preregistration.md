# Preregistration: native relevance of canonical route subspaces

The canonical route projector is stable under pairwise reconstruction, but that
comparison could be internally consistent and still irrelevant to the complete
model.  Use exactly the twelve blind failures and their frozen controls.

For every declared three-attention-layer route, compare its rank-16 unit-input
Fisher projector with the rank-16 projector of the complete native suffix
Jacobian (all attention, MLP, residual, normalization, and unembedding
derivatives).  Compare projector overlap and exact 5%-norm finite effects when
the current Layer-8 state is removed along each subspace.

At every site, also draw 32 seeded Haar-random rank-16 input subspaces.  Apply
the identical projection-removal and 5%-norm scaling, and compare exact KL.
Random subspaces are a sensitivity/null baseline, not a semantic control.

This is deliberately capable of invalidating the previous positive result: if
route/full overlap and finite fidelity fail, the canonical route object is a
stable surrogate but not a compact representation of the full local model. A
full subspace that fails to beat random would instead undermine the entire
local sensitivity-eigenspace interpretation or expose a finite-intervention
mismatch.
