# Layer-16 causal-basis red-team preregistration

This experiment was locked after the first exact-Jacobian pilot and before inspecting these controls.

The pilot found a discovery-frozen dense sensitivity basis that reproduced most of a cyclic Layer-16 edit's effect on the Layer-17 tensor operator. That is not yet evidence for a newly discovered causal code. The output is defined through a known 16-dimensional context span, so the Jacobian basis could merely rediscover that span. It also remains unknown whether native residual coordinates perform comparably, whether the result transfers to unseen sequence positions, and whether it depends on measuring tensor-operator change rather than raw coefficient change.

The red team therefore compares:

- the metric-Jacobian sensitivity basis;
- the same construction under Euclidean coefficient geometry;
- the frozen context span used to define the operator coefficients;
- the earlier predictive ridge span;
- top native residual coordinates ranked by discovery sensitivity;
- matched random dense and random native-coordinate controls.

All bases are frozen from a disjoint FineWeb discovery split. Evaluation covers disjoint FineWeb, WikiText, and TinyStories at both the four discovery position indices and four unseen position indices. Every projected edit is passed through the exact final attention and RMS normalization. Effects are scored both as raw coefficient changes and as changes weighted by the contracted tensor-operator Gram matrix.

Central finite differences across five step sizes separately test whether the autodiff Jacobian is numerically trustworthy. The locked hypotheses and thresholds are recorded in `configs/bilin18_layer16_causal_basis_redteam.json`.
