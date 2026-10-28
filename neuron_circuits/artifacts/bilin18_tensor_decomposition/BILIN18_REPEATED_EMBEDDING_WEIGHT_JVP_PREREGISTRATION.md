# Repeated-embedding weight-path JVP preregistration

The finite source factorial showed that repeated `x0` injection explains most of a current-token substitution, while the initial state and persistent Layer-0 value bus are individually weak. This follow-up replaces coalition attribution with a direct derivative of the model's weights and connections.

The current-position embedding input is represented as eight independently perturbable aliases:

- initial residual;
- persistent value-bus source;
- six copies injected into consecutive three-layer groups, 0–2 through 15–17.

At the native text state, forward-mode automatic differentiation computes an exact JVP for the donor-token direction through each alias. The path JVPs must sum to the simultaneous JVP. Their projection shares and subset fidelities are evaluated in the Layer-17 tensor-operator metric, the native-distribution Fisher metric on final logits, and the true-token CE derivative.

Finite edits at 0.01×, 0.1×, and 1.0× test the radius over which the weight-derived linear explanation remains valid. This is still state- and endpoint-conditioned: it is a direct contraction of the weights, but not a context-free ranking of parameter magnitudes.

The hypotheses and thresholds are locked in `configs/bilin18_repeated_embedding_weight_jvp.json`.
