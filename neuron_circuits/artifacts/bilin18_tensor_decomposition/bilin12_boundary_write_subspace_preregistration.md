# Preregistration: compact boundary-write module and state router

The fixed newline ray captures most write energy on FineWeb and TinyStories but
fails on WikiText, and per-example scalar routing does not repair it.  Therefore
the next hypothesis is a small bank of reusable write directions whose
coefficients rotate with context.

Fit an uncentered SVD to exact per-example Layer-11 token-to-newline write changes
on the FineWeb discovery panel.  Freeze ranks 1, 2, 4, 8, 16, 32, and 64.  On each
held-out corpus, first project the exact write change onto each fixed subspace;
this is an oracle coefficient upper bound.  Compare with an equal-rank random
subspace using the same oracle projection.

At the primary rank 16, fit a multivariate affine ridge router from the paired
change in normalized pre-MLP state to the subspace coefficients.  Select the
penalty on the separate FineWeb calibration panel, freeze it, and compare against
a shuffled-target router.  Evaluate finite patches through final RMSNorm and the
complete vocabulary softmax-Fisher metric.

Oracle failure falsifies a compact *global* boundary-write module at that rank.
Oracle success with router failure points to missing nonlinear/contextual routing.
Both successes would support a sparse operator-bank ontology, without implying
native-neuron alignment.
