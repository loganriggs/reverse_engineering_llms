# Preregistration: state contraction of the natural newline write ray

The fixed paired and unpaired newline edits were nearly collinear and behaved
similarly, but both lost fidelity on WikiText.  This assay asks whether the
direction is stable while its coefficient depends on the current computation.

For each token-to-newline pair, project the exact Layer-11 write change onto the
frozen natural-newline write ray.  The resulting per-example scalar supplies an
oracle upper bound.  Then predict that scalar from the change in normalized
pre-MLP state using affine ridge regression trained only on the FineWeb semantic
discovery panel.  Select the ridge penalty by scalar MSE on the already separate
FineWeb calibration panel, freeze it, and evaluate on held-out FineWeb, WikiText,
and TinyStories.

A shuffled-target ridge model with the same selected penalty controls for
high-dimensional fitting and generic coefficient statistics.  The causal endpoint
is full-vocabulary softmax-Fisher fidelity after finite write patches, not scalar
R2 alone.

Success would instantiate `operator + state contraction + effect`: a fixed late
write verb whose amplitude is computed from upstream state.  Failure of the oracle
would instead show that the effect direction itself rotates; oracle success with
predictor failure would imply a missing nonlinear or contextual routing variable.
