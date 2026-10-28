# Repeated-embedding path factorial preregistration

The tensor transformer injects the normalized token embedding `x0` into every block, and its attention stack also retains a persistent value bus initialized in Layer 0. Therefore, strong predictability from the current-token embedding and weak ablation of only the final direct injection do not establish that embeddings are merely correlational. The carried Layer-16 state is downstream of seventeen earlier `x0` injections.

This experiment replaces the current-position embedding with another real token embedding and separates three architectural source paths:

1. the initial residual state;
2. the persistent Layer-0 attention value bus;
3. the repeated `x0` injection supplied to every block.

All eight source coalitions are executed exactly. The all-native and all-donor endpoints must reproduce ordinary model execution. Coalition effects are scored against the complete donor-token substitution in two different geometries: the Layer-17 tensor-operator metric and the native-distribution Fisher metric on final logits. True-next-token cross-entropy effects are an additional scalar behavioral check.

The repeated-injection path is then divided into six consecutive three-layer groups. Isolated groups, cumulative prefixes, and cumulative suffixes are compared with the all-layer injection effect while initial state and value bus remain native. This is a source-path intervention, not a claim that the hybrid states are natural text.

The hypotheses and thresholds were fixed in `configs/bilin18_repeated_embedding_path_factorial.json` before running the experiment.
