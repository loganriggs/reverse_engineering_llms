# Preregistration: full-effect calibration of the grammar answer verb

Margin-calibrated generic answer patches reproduce the target class at Layers 1
and 2 but have poor full-distribution fidelity. Because calibration optimized
only one old-versus-new contrast, that failure could be an objective mismatch.

For every semantic transition and layer 1–3, compute the exact semantic patch
effect on discovery identities/template 0. Evaluate the frozen grid of scalar
multiples of the generic answer delta and select the scalar with highest median
softmax-Fisher fidelity to the complete semantic effect. Freeze it and evaluate
on held-out identities/template 6.

This is a strong equivalence test: one scalar may change magnitude but cannot
rotate the answer operator, condition on the recipient, or inspect evaluation
effects. Passing means the layer's semantic patch belongs to the same reusable
effect ray as the generic answer verb. Failure with adequate discovery fidelity
means either the direction or its scaling is state-specific. Layer 0 remains
excluded because its semantic patch is behaviorally negligible.
