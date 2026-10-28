# Clean-neuron scaling study

## Bottom line

Neuron cleanliness is not one-dimensional. Scaling can simultaneously make the trained neuron basis more privileged than a random rotation **and** make a learned distributed subspace much more efficient than individual neurons.

Across 104 independently trained models (13 configurations × eight seeds), every run converged. Confirmation accuracy was 100% in every condition mean except 10 classes / 256 neurons, whose mean was 99.96% because one seed scored 99.69%. Capacity failure therefore does not explain the trends.

## Metrics

The preregistered **Clean Axis Index** is the raw-neuron advantage over random rotated coordinates divided by the total advantage available from a learned PCA subspace, integrated over budgets from 1/64 to 1/4 of layer width. Zero means no privileged neuron basis; one means the raw basis is as efficient as the fitted subspace.

It must be interpreted alongside:

- **Raw-axis advantage:** raw minus rotation KL recovery, an absolute axis-alignment effect.
- **Threshold efficiency:** PCA dimensions divided by raw-neuron dimensions needed for 80% recovery. One is neuron-clean; small values mean the computation has a compact but distributed basis.

## One-factor scaling result

Final-layer Clean Axis Index means:

| Factor | Values | Operation counterfactual | Class counterfactual |
|---|---|---|---|
| Classes | 10 → 30 → 100 | 0.005 → 0.066 → 0.220 | -0.020 → 0.074 → 0.204 |
| Operations | 2 → 4 → 8 | 0.059 → 0.066 → 0.063 | 0.036 → 0.074 → 0.072 |
| Layers | 1 → 2 → 4 | 0.047 → 0.066 → 0.153 | 0.120 → 0.074 → 0.181 |
| Neurons | 64 → 128 → 256 | 0.087 → 0.066 → 0.072 | 0.100 → 0.074 → 0.050 |

The number of operation types had essentially no effect between two and eight. Class complexity had the largest and clearest association with neuron-axis alignment. Four-layer models were cleaner at their final layer, especially for operations.

Within the four-layer models, final-minus-first-layer Clean Axis Index was `+0.110` for operations (95% CI `[0.072, 0.148]`) and `+0.063` for classes (`[0.002, 0.124]`). This supports increased late-layer axis alignment, although proximity to the readout remains a possible mechanism rather than proof of clean internal communication.

## Class-count × width red-team

The preregistered 3×3 factorial showed that the positive class-count trend was not confined to one width. A seed-block bootstrap regression on log scale found:

| Endpoint | Family | log(classes) coefficient | 95% CI | log(width) coefficient | 95% CI |
|---|---|---:|---:|---:|---:|
| Clean Axis Index | Operation | +0.081 | [0.067, 0.092] | -0.017 | [-0.036, 0.005] |
| Clean Axis Index | Class | +0.061 | [0.047, 0.076] | -0.030 | [-0.060, -0.002] |
| Raw-axis advantage | Operation | +0.0246 | [0.0199, 0.0287] | +0.0123 | [0.0057, 0.0194] |
| Raw-axis advantage | Class | +0.0197 | [0.0152, 0.0241] | +0.0038 | [-0.0035, 0.0111] |

Increasing class complexity therefore produces more axis alignment, contrary to the simplest prediction that more tasks at fixed width must monotonically create more superposition.

Width exposes why a single metric is inadequate. The 80%-recovery threshold-efficiency coefficient for log(width) was `-0.203` for operations and `-0.202` for classes, with both confidence intervals entirely negative. Wider models often had a larger absolute raw-axis advantage, but their PCA subspaces became even more efficient relative to raw neurons. They were simultaneously more axis-aligned and more distributively compressible.

## Interpretation and limits

The best current account is:

1. More output distinctions encourage some trained axes to specialize, producing a real raw-neuron advantage over rotations.
2. Most causal structure still does not align with individual neurons: Clean Axis Index remains at most about 0.22, far from one.
3. Added width gives the model room for a compact distributed subspace without forcing that subspace onto coordinate axes.
4. Greater architectural depth concentrates axis alignment toward the final layer, but this task does not establish that additional algorithmic steps caused it.

This falsifies both extreme stories. The representations are neither arbitrary rotated clouds nor collections of fully clean neurons. They exhibit partial neuron specialization embedded inside a substantially cleaner distributed subspace.

## Files

- [One-factor visualization](clean_axis_scaling.png)
- [Class × width factorial](load_factorial.png)
- [One-factor statistics](summary.json)
- [Factorial statistics and bootstrap models](load_factorial_summary.json)
- [Initial preregistration](preregistration.md)
- [Factorial follow-up preregistration](load_factorial_preregistration.md)

