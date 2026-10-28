# Clean-axis scaling study preregistration

Sealed on 2026-10-17 before any scaling-grid run.

## Primary question

How does neuron-axis cleanliness change as we scale the number of classes, number of computation-like operations, network depth, and neuron width?

## Frozen design

Base condition: 30 classes, four cyclic-shift operations, two residual self-bilinear layers, and 128 neurons per layer. One factor changes at a time:

- classes: 10, 30, 100;
- operations: 2, 4, 8;
- layers: 1, 2, 4;
- neurons per layer: 64, 128, 256.

This gives nine unique configurations. Every configuration uses eight seeds, 32 training samples and eight independent confirmation samples per class-operation cell, cross-entropy training, five rotation nulls, decoder biases, and residual streams. All seeds and failed convergence runs are retained.

Implementation hashes:

- experiment: `5b64704e7578a0da4a0f055e645735d73da4bb423b12d5863acb6202829db285`;
- sweep: `05fd8067959b9780a75e4ca179e43befac1e4824acbd01e48edd8b592c7fbdcf`;
- aggregation: `af5e2055cf56b7751595a622be636f80bc8f3eee0bbdd32f833961c8cd32655f`.

## Frozen metrics

At fractional neuron budgets from 1/64 through 1/4 of layer width, let `R_raw`, `R_rot`, and `R_pca` be confirmation-set donor KL recovery. The primary metric is:

```text
Clean Axis Index = sum(R_raw - R_rot) / sum(R_pca - R_rot)
```

- zero means raw neuron axes confer no advantage over a random basis;
- one means raw neurons capture all sparse structure found by the learned PCA subspace.

Secondary metrics are raw-axis advantage, PCA-minus-raw advantage, and threshold efficiency `k_PCA / k_raw` at 80% and 90% KL recovery. All are reported separately for class and operation counterfactuals and for the first and final layer.

The primary plots use the final layer and show seed-level means with 95% normal confidence intervals. Confirmation accuracy and convergence are mandatory capacity controls.

## Competing predictions

- Superposition predicts decreasing cleanliness with more classes or operations at fixed width, and increasing cleanliness with width at fixed load.
- Clean internal communication predicts greater cleanliness in later/deeper computation, particularly for reusable operations.

No condition, layer, or counterfactual family will be removed based on its result.

