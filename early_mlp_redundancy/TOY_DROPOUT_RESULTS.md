# T4: the toy robustness tradeoff can be reduced by the objective and capacity

2026-10-28. Completes the fixed toy grid: nine new models and three reused controls, with no
hyperparameter search. Same16-vector task, input noise.25, contribution penalty.01, three seeds,
Adam.01,2000steps. Width16/32 crossed with training-time channel dropout0/.25. Masks remove exactly
a quarter of channels per batch, without rescaling; all channels are used at evaluation. Dropout
has an independent generator, preserving paired input streams. Unmasked contributions receive the
same energy penalty. See the [pre-outcome protocol](TOY_DROPOUT_PROTOCOL.md).

## Both perturbation types can be tolerated by the classification metric

Means across all three seeds; all models have100% intact clean classification accuracy.

| Width | Training channel dropout | Clean vector error | Accuracy at input noise.25 | Accuracy after random25% channel deletion | Class accuracy under its own worst single-channel deletion |
|---|---:|---:|---:|---:|---:|
|16|0|.0664|97.82%|77.42%|7.00%|
|16|25%|.0679|98.01%|80.42%|49.15%|
|32|0|.0676|97.66%|94.61%|85.06%|
|32|25%|.0345|97.36%|98.36%|100%|

The last column allows a different worst deleted channel for each class, not a single common
intervention. Width32 with dropout preserves every clean class decision under every tested single
channel deletion in all three seeds. This does not imply exact vector preservation, robustness
to arbitrary multi-channel deletion, or a population guarantee beyond these balanced draws.

The average number of classes with any single-channel error(out of16) falls15→9.33 at width16,
and3→0 at width32. This shows the T3 vulnerability is not unavoidable in this toy family.

Capacity is not free: width16 has1072parameters, width32 has2128. Initial weight distribution is
held fixed, but initial functions differ across widths. Within a width, dropout/no-dropout use
identical initial tensors and paired data streams. Comparisons are descriptive over three seeds,
not a statistically certified frontier or a proof that this is the best training objective.

## Robust classification is not unchanged computation

| Width | Dropout | Vector error at input noise.25 | Error at input noise.5 | Error after25% deletion |
|---|---:|---:|---:|---:|
|16|0|.2403|1.2436|.3023|
|16|25%|.2431|1.2885|.2989|
|32|0|.2414|1.2414|.2634|
|32|25%|.2625|1.5883|.2007|

The width32/dropout solution worsens noisy-output reconstruction despite strong label accuracy.
Unrescaled dropout can change intact output calibration; no post-hoc gain repair was applied.
Input-noise.5 accuracy stays around55% in all four conditions, so increased capacity/dropout does
not solve severe ambiguity. The result is improved joint classification robustness, not equal
fidelity under every metric or a free robustness gain.

## Implementation shifts

| Width | Dropout | Channel/net energy | Contribution participation | Class-energy specificity |
|---|---:|---:|---:|---:|
|16|0|1.010|1.63|.986|
|16|25%|.971|6.48|.815|
|32|0|.608|3.69|.902|
|32|25%|.500|10.68|.807|

Dropout spreads contribution energy across more channels and reduces net destructive interference.
Class specificity remains substantial: meaningful class selectivity and distributed robustness
can coexist. “Specialized” and “redundant” therefore need not be opposite layer categories.
Participation is not feature rank or a count of independent computations. Neither the small grid
nor these correlations prove mediation of the accuracy change by one statistic.

There is a useful capacity boundary: removing25% of width16 leaves12 output directions, below
the15-dimensional target span, whereas removing25% of width32 leaves24. Thus exact vector fitting
after every such mask is already rank-obstructed in the smaller case. This is not a classification
impossibility theorem, and does not prove that rank alone explains the optimization outcomes.

## Close this toy grid and return to real-model signatures

This fixed comparison finishes the planned grid. No additional dropout/noise/width sweep follows.
The learned toy results establish possibilities and conditional tradeoffs, not the LLM's training
history or a universal naming scheme. Use [the synthesis](STRATEGY_SYNTHESIS.md) to map what is
supported and what remains missing.

The next bounded real-model diagnostic should measure the function's empirical output spectrum
alongside channel coordination for MLP0/1/2 on the same already-opened inputs. Compare the observed
contribution energy with the nuclear-norm lower bound, report occurrence-weighted and deduplicated
causal-prefix views, and preserve sample limitations. Such a bound concerns free rank-one matrix
factorization over the measured data: it is NOT proof of an executable lower-width bilinear MLP,
which has additional constraints on activation functions. Do not fit or prune from this diagnostic.
This tests whether the toy's effective-rank/implementation distinction is even a useful description
of the real layers before proposing a new featurizer objective. Natural-text generalization and
whole-task holdout transfer remain open.

## Reproduction and audit

```bash
python train_dropout_toys.py --output validation/dropout_toys.json
python audit_dropout_toys.py
```

The trainer requires T3's `/tmp/emlp_t3_models.pt` for controls and writes `/tmp/emlp_t4_models.pt`.
Independent NumPy replay covers all12models:36 clean/noisy evaluations,192 quarter-deletion
evaluations and288 exhaustive single-channel evaluations(516 total). Confusion counts match;
maximum forward/metric errors1.28e-6/1.01e-7. Geometry and targeted summaries checked. The three
reused controls replay T3 exactly. [Results](validation/dropout_toys.json),
[audit](validation/dropout_toys_audit.json), [parameter weights](validation/dropout_toy_weights.json).
