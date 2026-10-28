# Red-team audit and a falsifiable clean-neuron experiment

## Bottom line

The current experiments support three claims: residual transmission materially improves optimization reliability; diverse computation queries can force held-out rule generalization; and circuits that preserve correctness are much smaller than circuits that preserve cross-entropy confidence. They do **not** yet provide strong evidence that individual raw neurons are clean, monosemantic computational units.

The strongest apparent cleanliness result is largely explained by the experimental interface and by representation-level structure that survives random rotation. The principled next experiment should treat clean neurons, clean subspaces, and distributed computation as competing hypotheses and use held-out causal tests to distinguish them.

## Red-team findings

### 1. The candidate “verbs” are explicitly supplied to the model

The input contains one dedicated coordinate for each query arity. Among the 72 top-eight neurons from the three maximum-load models, 59 were best explained by an explicit arity coordinate and 13 by the target label. Detecting a supplied task marker can be useful routing, but it is not evidence that the model discovered the parity algorithm or a reusable internal verb.

The target-label correlations are also ambiguous: they may reflect an answer representation downstream rather than a neuron with a coherent input-side meaning.

### 2. The cleanliness score is not specific to the learned neuron basis

For the saved seed-7 model, the fraction of units above the current cleanliness threshold was compared with 100 random orthogonal rotations of the same learned activations and 20 untrained models:

| layer | raw learned neurons | random rotations, mean ± SD | untrained, mean ± SD |
|---:|---:|---:|---:|
| 1 | 0.047 | 0.031 ± 0.021 | 0.071 ± 0.044 |
| 2 | 0.031 | 0.167 ± 0.040 | 0.143 ± 0.048 |
| 3 | 0.703 | 0.648 ± 0.045 | 0.159 ± 0.054 |

Training creates strong late-layer feature structure, but most of the layer-3 score remains after arbitrary rotation. Layer 2 is actually less “clean” in its raw neuron basis than its rotations or untrained controls. This supports a representation- or subspace-level claim, not a raw-neuron claim.

### 3. Neuron counts are not intrinsic properties of the computed function

The model has 192 per-input bilinear activations but 76,692 parameters. Fact identity is represented by one input coordinate per fact, so parameter count grows with the number of facts. “Facts per neuron” is therefore a width-sharing statement, not a storage-capacity result.

A bilinear write is a tensor decomposition

`sum_i d_i (l_i^T x)(r_i^T x)`.

Such decompositions admit scaling, permutation, duplication, splitting, and sometimes more general non-uniqueness. A functionally identical wider model can clone one neuron into several and change both cleanliness and circuit-size counts. Any raw-neuron claim needs function-preserving reparameterization and cloning controls.

### 4. Existing circuit sizes are heuristic upper bounds

The masks are selected by intact `|activation × gradient|`, tested on the same task-family sample used to aggregate importance, and searched over a coarse count grid. Ablation effects are non-monotonic, and activation-times-gradient is only a local approximation. The reported counts are therefore “smallest count found by this ranking and grid,” not minimal sufficient circuits.

Input-specific masks are additionally answer-informed and can encode information through the mask pattern. They measure conditional compressibility, not a deployable circuit.

For a clean test, mask discovery, hyperparameter choice, and final evaluation must use disjoint data. Learned L0 masks, greedy deletion, and the current ranking should be compared under the same validation protocol.

### 5. Accuracy and CE measure different—and unstable—properties

The 20-seed study validates the concern about softmax confidence. The full model’s median held-out 8-fact CE is `1.03e-4`, but its mean is 1.35 with SD 4.36. Two of twenty full models have 8-fact CE worse than chance, and one has 98.7% accuracy but CE 34.97 because a few mistakes are extremely confident.

CE-based circuit sizes consequently depend on arbitrary logit scale, training duration on separable data, and rare tail errors. They are useful for diagnosing confidence, but should not be interpreted as an intrinsic amount of computation without temperature calibration, margin normalization, or explicit norm regularization.

### 6. Training convergence is not rule learning

All 20 residual/no-bias models met the training convergence criterion, but their held-out 8-fact accuracy was `0.915 ± 0.184`. Several learned the training distribution without learning the general rule. Conversely, the strict convergence flag can fail for a model that is nearly correct.

Train loss, full-train accuracy, held-out accuracy, held-out CE, and algorithmic counterfactual consistency should be separate endpoints.

### 7. Residual and bias ablations change more than “communication cleanliness”

Residuals and biases alter optimization, signal scale, and the available polynomial degrees. With neither, the network is a deep homogeneous polynomial and default initialization drives activations and gradients toward zero. Its failure at `log(2)` is strong evidence of an optimization pathology under this parameterization, not proof that residual streams are required for the function.

A fair expressivity claim needs analytic representability checks plus separately tuned, variance-preserving initialization for each architecture. A fair optimization claim should retain the matched initialization and explicitly label the conclusion as optimization-specific.

### 8. Seed variation currently mixes several sources

Each seed changes parameters, fact values, and sampled query sets simultaneously. This estimates end-to-end reproducibility, which is useful, but it cannot say whether variance comes from initialization, dataset composition, or the stored facts. A factorial replication should vary these independently.

Circuit statistics are conditional on the 19/20 full models that converged, while architecture averages include failures. That conditioning must remain explicit.

## A precise clean-neuron hypothesis

“Some neurons look correlated with concepts” is too weak to falsify. A useful strong hypothesis is:

> Under pressure to reuse computations across many tasks, a substantial fraction of the model’s **causal computational effect** is carried by individual raw neurons whose activation condition and output role admit one short, stable explanation that predicts held-out counterfactual interventions better than matched distributed-direction controls.

This separates five requirements:

1. **Activation selectivity:** one latent factor predicts the neuron on held-out data.
2. **Nuisance invariance:** other independently varied factors add little predictive power.
3. **Causal specificity:** patching the neuron changes the behavior predicted by its explanation and few unrelated behaviors.
4. **Causal importance:** the neuron has a nontrivial effect; a selective but irrelevant neuron does not count.
5. **Input-output agreement:** the same explanation predicts both when the neuron activates and why its decoded write is useful.

Monosemanticity is the conjunction. No single correlation or decoder direction is sufficient.

## Proposed confirmatory benchmark

### Data-generating process

Use a factorial latent-variable benchmark with known factors:

- fact identity and fact value;
- requested operation;
- operand role and number of operands;
- algorithmic stage or intermediate result;
- answer;
- nuisance variables with matched marginal statistics.

Generate paired counterfactuals that change exactly one factor while holding all others fixed: operation swaps, one-value flips, operand swaps, irrelevant-fact changes, and matched-answer pairs. Do not make the main explanatory variable a dedicated scalar input. Compare explicit one-hot instructions, random distributed instruction codes, and instructions that must be inferred from context.

Include multiple operations with matched inputs and balanced answers: parity/product, max, min, second maximum, comparison, and bracket-state transitions. Include random-label memorization controls. Parity alone is unusually aligned with multiplicative bilinear layers.

### Experimental factors

Train at least 20 seeds while independently crossing:

- computation-task diversity at fixed example count;
- memory load at fixed parameter count;
- width and depth;
- standard MLP versus bilinear layers;
- communication bottleneck, activation noise, and sparsity pressure;
- explicit versus inferred task instructions.

Hold fact values and datasets fixed while varying initialization, then separately vary datasets and facts. This decomposes variance rather than mixing it.

### Three-way data split

1. **Discovery:** identify candidate neurons and write an explanation.
2. **Intervention design/validation:** choose thresholds and patch types.
3. **Locked confirmation:** evaluate the frozen explanation, neuron, mask, and thresholds.

No neuron ranking, explanation editing, or threshold selection may use the confirmation set.

## Measurements

Do not collapse everything into one scalar initially. Report a neuron “certificate” with separate gates:

### Activation meaning

- Cross-validated single-factor prediction of activation (`R²`, likelihood gain, or AUC as appropriate).
- Incremental gain from all other factors after the proposed factor is known.
- Consistent response to matched factor-changing counterfactuals.
- Invariance to matched nuisance counterfactuals.

### Causal role

- Natural donor patches, zero/mean ablations, and activation resampling.
- Necessity and sufficiency on locked examples.
- An effect vector over every task and output contrast, not only the contrast used to discover the neuron.
- Interaction terms from joint patches, because bilinear computations need not be additive.

### Input-output agreement

From activation examples alone, predict the sign and family of output effects. From decoder weights alone, predict activation contexts. Score both predictions on held-out counterfactuals. A clean neuron should pass in both directions.

### Model-level endpoints

- **Certified clean-neuron fraction:** fraction of causally nontrivial raw neurons passing every gate.
- **Clean causal coverage:** fraction of held-out behavioral effect reproduced by the complete set of certified neurons.
- **Distributed advantage:** performance difference between the best `k` raw neurons and a `k`-dimensional learned subspace with the same selection budget.
- **Basis advantage:** raw-neuron score minus random-rotation and magnitude-matched direction nulls.
- Accuracy, normalized margin, calibrated CE/KL, and worst-group/tail loss.

Use group interventions or approximate Shapley decompositions to report interaction residuals; summing individual effects can otherwise double-count causal influence.

## Null controls that can falsify the raw-neuron claim

1. Random orthogonal rotations of each layer.
2. Magnitude- and sparsity-matched random directions.
3. Sparse learned subspaces with the same dimensionality and discovery budget.
4. Untrained networks and shuffled semantic labels.
5. Random counterfactual donors matched for activation magnitude.
6. Function-preserving neuron cloning, splitting, and rescaling.
7. Decoder-only and input-marker-only baselines.
8. Discovery-label permutations to estimate winner’s-curse inflation.

If raw neurons do not outperform these controls, the correct conclusion is clean representation geometry or a clean subspace—not clean neurons.

## Pre-registered decision rule

An illustrative strong claim, to fix before seeing confirmatory results, is:

> At high computation diversity, at least 50% of held-out causal effect is covered by raw neurons that each achieve activation selectivity, counterfactual consistency, causal specificity, and input-output agreement of at least 0.8; raw-neuron clean coverage exceeds the 99th percentile of rotation controls and improves by at least 0.10 over the low-diversity condition.

Use seed-level bootstrap intervals and a predeclared equivalence margin. Falsify the strong clean-neuron hypothesis if any of the following holds:

- the 95% upper confidence bound on clean causal coverage is below 50%;
- the 95% upper confidence bound on the high-minus-low diversity effect is below 0.10;
- raw neurons fail to beat rotation or matched-direction controls;
- a low-dimensional subspace succeeds while no individual units pass the causal certificate;
- explanations fail on locked counterfactuals or require multiple unrelated factors;
- the claimed neuron count changes drastically under function-preserving cloning/splitting.

These failures would not imply that the model lacks interpretable structure. They would distinguish the stronger raw-neuron hypothesis from cleaner and more plausible alternatives: interpretable subspaces, distributed algorithms, or context-dependent ensembles.

## Recommended next experiment

The highest-value next step is not another larger accuracy sweep. It is a preregistered **raw neurons versus rotated directions versus learned subspaces** experiment on a factorial multi-operation benchmark, with matched counterfactual patches and a locked confirmation set. A 10-/100-class version is specified in `artifacts/multiclass_clean_neurons/design.md`. That experiment directly tests whether neuron boundaries are privileged and whether a single explanation predicts both activation and causal output role.
