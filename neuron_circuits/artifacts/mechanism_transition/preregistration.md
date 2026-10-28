# Mechanism-transition experiment

## Status

The seed-0, load-128 optimizer comparison is exploratory because it motivated this locked follow-up. It used a four-layer width-96 bilinear grammar transformer with answer-only loss. SGD and AdamW both passed the factual/template gate. The local signed kernel predicted one-logit finite edits essentially exactly, but SGD did not show a stronger web: pooled block collateral was 0.0142 logits for SGD and 0.0217 for AdamW.

No claim about saturation or optimizer-by-saturation interaction will be made from that single seed.

## Locked primary follow-up

Use the same generator, four-layer width-96 bilinear transformer, six training templates, answer-only loss, and fully arbitrary labels. Vary:

- training facts per relation/answer stratum: 64, 128, and 256;
- optimizer: AdamW and momentum SGD;
- independent data/model seeds: 0, 1, and 2.

Use learning rate 0.001 for AdamW and 0.03 for SGD. Train until the existing gate or 40,000 steps. A condition enters the primary geometry comparison only when full-panel fixed-template accuracy is at least 0.97 and unseen-template accuracy is at least 0.95. Failed cells remain behaviorally informative but are not used as evidence about the geometry of learned facts.

## Measurements

On a deterministic 64-fact panel, freeze the strongest incorrect competitor and compute exact gradients of correct-versus-competitor margin for input parameters, each complete transformer block, and the output group.

Primary endpoints, averaged over complete transformer blocks:

1. controlled mean absolute off-diagonal gradient cosine;
2. mean absolute finite collateral after an edit normalized to predict a one-logit source-margin increase;
3. effective support of absolute signed coupling.

Report baseline margin quantiles and gradient-norm quantiles. The signed-kernel predictor must achieve Pearson at least 0.8 and origin-anchored R-squared at least 0.5 before its collateral is interpreted.

Add a nonnegative KKT reconstruction audit after the causal endpoints are complete. Fit source-margin gradients to the parameter vector separately for MLP parameters and whole blocks. Report held-out reconstruction cosine and multiplier sparsity. This is descriptive unless the relevant homogeneous/max-margin assumptions can be justified.

## Hypotheses and falsification

- **Saturation hypothesis:** at matched fidelity, increasing fact load increases coupling magnitude and normalized collateral within both optimizers.
- **SGD interaction hypothesis:** the load slope is larger for SGD than AdamW.
- **Diffuse-slack hypothesis:** effective support can rise while coupling magnitude remains weak when the model has spare capacity.

The saturation hypothesis is falsified for this substrate if neither optimizer shows a positive paired load trend in coupling or collateral across the three seeds. The SGD interaction is falsified if its paired slope is not larger than AdamW's. A failure to fit high-load cells is a capacity/optimization result, not evidence for a taut learned web.

If optimizer and load do not explain the transition, the next locked factorial changes depth/residual structure and normalization while retaining answer-only loss and matched behavioral gates.
