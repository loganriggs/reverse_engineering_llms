# Preregistration: semantics of the stable tangent-bundle output core

The role-separated final-MLP tensor has a more stable output core than context
or tangent cores. This experiment asks whether that stability corresponds to
human-auditable output roles or merely a reproducible numerical subspace.

Three claims are kept separate:

1. **Weight-only vocabulary geometry:** fold each physical output mode through
   the unembedding and ask whether it separates preregistered token properties.
2. **Natural output questions:** project exact target-token and model-top1
   residual adjoints onto the modes, select simple explanations on FineWeb,
   and evaluate them on disjoint FineWeb, WikiText, and TinyStories.
3. **Axis privilege:** repeat the complete selection procedure after random
   orthogonal rotations of exactly the same physical rank-16 subspace.

Token properties comprise the existing fourteen structural flags, six generic
shape flags, and ten mutually exclusive coarse lexical classes. Vocabulary
selection uses even token IDs and evaluation uses odd IDs. Natural explanation
selection uses the first half of a separate 16,000-site FineWeb panel; its
second half is a confirmation set. The larger evaluation panels are disjoint.

Success of the full subspace decoder with failure of axis privilege means the
semantic information is real but not localized to the HOSVD coordinates.
Success of individual axes against rotations supports a stronger canonical
output-verb claim. Top activation examples remain descriptive; all claims use
held-out AUROC or decoder transfer.

Locked thresholds and data ranges are in
`configs/bilin18_layer17_tangent_output_semantics.json`.
