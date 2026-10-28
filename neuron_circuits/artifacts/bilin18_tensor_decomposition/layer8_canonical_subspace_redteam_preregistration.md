# Preregistration: canonical sensitivity subspace red-team

Use exactly the twelve blind sparse-program failures and their already frozen
same-site controls.  Do not select new examples or masks.

For each exact three-layer route response and its singleton/pair Mobius
reconstruction, divide each response column by its signed natural mode
coefficient.  This converts literal-mode-removal columns into derivatives per
unit input displacement.  Exclude only coefficients below `0.001 ×` the site's
median absolute coefficient to avoid float32 division noise; report the count.

Diagonalize the resulting 512-mode Fisher Gram operators.  The primary object is
the rank-16 projector, which is invariant to sign, ordering, and rotations
inside a degenerate subspace.  Report normalized projector overlap, leading
vector cosine, eigengaps, and Gram error.  As an exact finite test, remove the
current state's projection onto each recovered rank-16 input subspace at a 5%
input-norm budget and compare the complete model's output effects.  Also compare
5%-norm edits along sign-aligned leading eigenvectors.

The key falsifier is direct: if projector overlap or the finite subspace effect
fails in the same cases as the sparse set, the instability belongs to the local
causal geometry rather than only to greedy discrete extraction.  Top-vector
failure is interpreted only when its eigengap is at least 1.1; otherwise the
projector is the canonical object.
