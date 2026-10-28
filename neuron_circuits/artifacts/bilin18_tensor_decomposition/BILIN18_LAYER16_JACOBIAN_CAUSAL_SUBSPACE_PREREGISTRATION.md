# Preregistration: exact Layer-16 Jacobian causal subspaces

The depth trace found a near-perfect linear predictor from Layer-16 state to
the final compact operator. A prospective intervention then falsified the claim
that the predictor's 16-dimensional Euclidean input span is the causal channel:
patching it explains only a small fraction of the full-state effect.

This follow-up replaces regression geometry with the exact local suffix
Jacobian from the selected Layer-16 current-position state to the sixteen
context coefficients. The output metric is not Euclidean coefficient norm; it
is the Gram matrix induced by the contracted tangent-to-output tensor, so the
stacked sensitivity rows are `G_operator^(1/2) J`.

Discovery Jacobians define nested aggregate sensitivity subspaces of ranks 16,
32, 64, and 128. Held-out cyclic state edits are projected into those spaces,
the failed predictive span, matched random spaces, and each site's own Jacobian
row space. Every candidate is inserted into the real Layer-16 state and the
complete final attention/RMS path is recomputed.

Both 0.1x and full cyclic edits are tested. The exact Jacobian is expected to be
a local causal explanation; degradation at full scale is evidence of curvature,
not a failure of differentiation. A site-specific row space is an oracle upper
bound. A shared sensitivity space must transfer across FineWeb, WikiText, and
TinyStories to count as a reusable intervention coordinate.
