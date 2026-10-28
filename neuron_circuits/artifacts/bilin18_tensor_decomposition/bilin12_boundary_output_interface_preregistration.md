# Preregistration: natural boundary computation versus generic output interface

The synthetic grammar experiments found that rule-specific evidence converges to
a codebook-relative late MLP write/effect interface.  This assay tests the same
ontology on an independently trained 12-layer, 307M-parameter-style tensor
language model without imposing synthetic labels.

At Layer 11, replace the current final-position token by the GPT-2 newline token.
The causal target is the exact per-example change in the final MLP residual write,
realized on the unedited recipient state and evaluated through final RMSNorm and
the complete vocabulary distribution.  This isolates the final-MLP path rather
than claiming to reproduce the entire token edit.

Three frozen candidate edits are learned on disjoint FineWeb ranges:

1. the paired mean write change caused by token-to-newline edits;
2. an unpaired natural newline-versus-non-newline write prototype, position
   matched over 16 sequence locations; and
3. an analogous period-token prototype as a semantic/control direction.

One scalar for each unpaired prototype is chosen on a separate FineWeb calibration
range to maximize full softmax-Fisher fidelity, then frozen.  Evaluation uses new
FineWeb rows plus WikiText and TinyStories.  A positive result means the late
boundary computation has collapsed onto a reusable output-effect interface; it
does not imply a single neuron, and it does not locate the upstream boundary
detector.
