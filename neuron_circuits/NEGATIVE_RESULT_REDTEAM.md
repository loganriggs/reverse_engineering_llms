# Red-team protocol for negative and falsifying results

## Principle

A negative result can mean either that the mechanism is absent or that the assay could not see it. We call a hypothesis falsified only after demonstrating that the complete pipeline would have detected the mechanism at a scientifically meaningful strength.

## Gate 1 — Define the exact claim

Before running:

- state whether the claim is universal, regime-specific, or quantitative;
- choose the minimum effect that would matter;
- lock the examples, metric, intervention scale, comparison budget, and exclusion rules;
- state which outcomes are falsification, support, or inconclusive.

Failure in one layer or task only falsifies a universal claim if the hypothesis actually predicted that layer or task.

## Gate 2 — Qualify the assay

Every proposed falsifier needs:

1. **Identity oracle:** the full representation/intervention reproduces the original result.
2. **Known-mechanism positive control:** run the pipeline on a model with an implanted mechanism of the target type.
3. **Injection recovery:** add weak mechanisms at several known strengths and estimate the detection threshold.
4. **Null calibration:** random labels, rotations, shuffled pairings, untrained weights, or matched nonsemantic controls as appropriate.
5. **Independent calculation:** where feasible, compare hooks with direct tensor contraction, JVPs with finite differences, or two separately written implementations.

If the assay misses the positive control or only detects unrealistically large injections, a real-model null is uninterpretable.

## Gate 3 — Match nuisance variables

Mechanistic comparisons require matching or conditioning on:

- task accuracy and held-out transfer;
- baseline margin and cross-entropy;
- activation/write magnitude;
- parameter and gradient norms;
- rank, energy, neuron count, and search budget;
- token, answer, competitor, and input overlap;
- intervention scale and numerical precision.

For optimization comparisons, also report training exposure, stopping rule, margin distribution, and whether the assumed implicit-bias/KKT conditions apply.

## Gate 4 — Stress the causal result

- sweep small through moderate intervention scales;
- retain positive and negative effects;
- recompute all downstream nonlinearities and normalization;
- check competitor switching and sign reversals;
- evaluate both local quantities and the final output distribution;
- separate discovery examples from transfer examples;
- use at least one OOD corpus when the hypothesis claims reusable language structure.

## Gate 5 — Establish power and replication

- use trained models/data seeds as replication units, not neurons or prompts;
- report confidence intervals against the minimum meaningful effect;
- replicate decisive negatives on fresh seeds;
- show per-seed and per-context distributions rather than only pooled means;
- audit the hardest failures manually with worked examples.

A small nonsignificant mean is “not detected.” It becomes falsification only when the interval excludes the locked meaningful effect and the assay qualification gates pass.

## Gate 6 — Adversarial alternative explanations

For every negative, explicitly test at least these alternatives:

1. the representation exists at another token position or layer;
2. the coordinate basis is wrong but a subspace explanation works;
3. the local metric is correct but the downstream suffix changes its importance;
4. the phenomenon is conditional on state and was averaged away;
5. the model failed to learn the target behavior;
6. numerical precision or normalization corrupted the intervention;
7. selection and evaluation reused the same examples;
8. the hypothesis predicts an equivalence class rather than a unique support.

## Negative-result vocabulary

- **Assay failure:** controls show the method cannot detect the target mechanism.
- **Inconclusive:** assay works, but power, fidelity matching, or replication is insufficient.
- **Disfavored in this regime:** locked effect is absent under the tested conditions.
- **Falsified as a universal claim:** a valid counterexample violates the hypothesis's stated universal prediction.
- **Retired:** repeated decisive tests leave the hypothesis dominated by a simpler alternative across its intended domain.

## Application to the current optimizer pilot

The seed-0 answer-only comparison did not show a stronger SGD constraint web. This is **not** yet a falsification of the saturation/KKT hypothesis:

- optimizer changed, but saturation, depth, residual paths, and normalization did not;
- only one seed was tested;
- KKT stationarity itself was not measured;
- both models may remain in a high-slack regime.

It does disfavor the narrow claim that switching AdamW to SGD alone necessarily produces a taut web in this four-layer substrate. The locked load × optimizer replication is the appropriate next test.
