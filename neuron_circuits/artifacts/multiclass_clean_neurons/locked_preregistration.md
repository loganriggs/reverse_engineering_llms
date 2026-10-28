# Locked confirmation preregistration

Sealed before any 100-class run on 2026-10-17.

## Question

Do trained bilinear neuron coordinates provide a privileged sparse basis for causal class and operation counterfactuals, or is the causal information substantially cleaner in a learned distributed subspace?

## Frozen implementation and configuration

- Experiment script SHA-256: `fab4deea2a94f77aeca7b64971bfaa8c6bb14849557d4c0893f3bc39df616f24`
- Model: two residual self-bilinear layers, `y = x + D((Lx) ⊙ (Rx)) + b`, 256 neurons per layer, decoder biases enabled.
- Inputs: 100 random dense class prototypes in 128 dimensions with Gaussian noise SD 0.5, plus eight distributed 16-dimensional orthogonal operation codes.
- Labels: exactly one of 100 classes; operation `j` maps class `c` to `(c+j) mod 100`.
- Training: cross-entropy, AdamW, LR 0.005, batch size 1024, at most 5000 steps, 32 samples per class-operation cell.
- Confirmation: 16 newly generated noisy samples per class-operation cell.
- Seeds: 0–9, all reported. A non-converged run remains in the report and is not silently replaced.
- Null: 10 random orthogonal bases per seed, each ranked by discovery-set delta energy exactly as raw neurons are.
- Comparison: raw neuron axes, random rotated axes, and PCA directions fitted on discovery activation deltas.
- Counterfactual families: operation changes and class changes, analyzed separately.
- Fixed ranks: 1, 2, 4, 8, 16, 32, 64, and 256. Rank 256 is a full-space implementation check and is excluded from sparse-curve statistics.
- Validation fixes the softmax temperature; confirmation is touched only for final measurement.

Exact command template:

```bash
uv run python scripts/run_multiclass_subspace_pilot.py \
  --input-mode prototypes --classes 100 --content-dim 128 \
  --operations 8 --code-dim 16 --neurons 256 --depth 2 \
  --train-samples-per-cell 32 --confirmation-samples-per-cell 16 \
  --max-steps 5000 --rotations 10 --seed SEED \
  --output artifacts/multiclass_clean_neurons/locked_confirmation100/seed_SEED
```

## Locked endpoints and decision rules

For each family and layer, compute over ranks 1–64:

1. **Axis privilege:** mean across ranks of `raw KL recovery - mean rotated KL recovery`.
2. **Distributed-subspace advantage:** mean across ranks of `PCA KL recovery - raw KL recovery`.
3. Smallest tested rank reaching 80% and 90% KL recovery for raw, rotation, and PCA; failure to reach is reported as `>256`.

Aggregate each endpoint across the 10 seeds with its mean, SD, and 95% normal CI.

- Strong support for privileged clean neuron axes requires the lower 95% CI of axis privilege to exceed `+0.10`.
- The practically substantial clean-axis claim is falsified for this setup if the upper 95% CI is below `+0.05`.
- Evidence for a distributed causal subspace requires the lower 95% CI of PCA-minus-raw to exceed `+0.10`.

These thresholds are effect-size rules, not merely significance tests. Results for both layers and both families will be disclosed; no favorable family or layer will be selected after the fact.

