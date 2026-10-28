# Preregistration: symmetry-aware MLP-write invariance

Raw post-product bilinear channels are not identifiable: channel permutations and
reciprocal rescalings can change their coordinates without changing the MLP's
residual write.  The same-codebook raw-channel alignment therefore may be asking
for a correspondence that the function does not possess.

This assay repeats the same leave-one-relation-out initialization comparison in
the MLP **write space**.  For each layer, natural channel activations are multiplied
by the frozen down-projection, producing residual writes.  A global affine ridge
map aligns source writes to target writes using only other relations.  A source
semantic activation edit is converted to its write, mapped, and realized in the
target model using the minimum-norm pseudoinverse of the target down-projection.
The reconstructed write must match numerically before any causal score is trusted.

This quotient removes channel null directions, permutations, and scale symmetries,
but residual-stream coordinates can still be initialization-specific.  Therefore
a positive result supports a more canonical operator/effect representation than
native neurons; a valid negative result would reject even this global-linear
write-space equivalence.  It would not exclude nonlinear alignment.
