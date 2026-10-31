# Reduction metric × legibility prior on trained two-head retrieval

The two-output retrieval toy is a useful transfer test because both trained attention heads are needed to solve the task. For each of ten independently trained checkpoints, we constructed the exact *combined* token-function table (four query/output rows by six key tokens), reduced it to rank two, and selected an orthogonal basis within that reduced function. The basis change preserves the rank-two function exactly; it does not rotate the physical heads or certify a unique circuit.

## Factorial design

The reduction metric was either uniform over all 24 token-function cells (weight-only, no sampled inputs), estimated from 100,000 unlabeled training-distribution sequences, or a 50:50 mixture. The frequency metric uses query/key occurrence counts, with no output labels. For each reduced function, we scanned 4,096 rotations and compared the SVD baseline with six objectives: key-dictionary L1, output-row L1, normalized sum of both factor L1s, component-response L1, mean L1 change across all 15 unordered key-token swaps, and mean L1 change across three *known-role* bit flips. The all-pair objective uses no role or task labels; the known-role objective is a positive control.

| Reduction | Gauge | Mean A/B alignment | Worst seed | Intended bit-effect fraction | Full / rare task accuracy |
|---|---|---:|---:|---:|---:|
| Uniform | SVD | .884 | .737 | .829 | 1.00 / 1.00 |
| Uniform | key L1 | .975 | .929 | .997 | 1.00 / 1.00 |
| Uniform | row L1 | .976 | .931 | .999 | 1.00 / 1.00 |
| Uniform | response L1 | .976 | .931 | .999 | 1.00 / 1.00 |
| Uniform | all-pair edit L1 | .976 | .930 | .999 | 1.00 / 1.00 |
| Uniform | known-role edit L1 | .975 | .929 | .999 | 1.00 / 1.00 |
| Empirical | SVD | .886 | .737 | .829 | 1.00 / 1.00 |
| Empirical | all-pair edit L1 | .978 | .953 | .999 | 1.00 / 1.00 |
| Half mixture | SVD | .886 | .738 | .830 | 1.00 / 1.00 |
| Half mixture | all-pair edit L1 | .977 | .945 | .999 | 1.00 / 1.00 |

The normalized two-factor L1 condition is similar: alignment .975/.978/.977 for uniform/empirical/mixed. The remaining conditions and per-seed results are in `dual_joint_edit_gauge.json`. All seven gauges preserve the selected rank-two table to numerical precision. All 30 rank-two reductions retain 100% joint task accuracy on the exhaustive test and on the rare-combination subset. Mean relative uniform table MSE is .00491 for uniform, .00737 for empirical, and .00559 for mixed.

The alignment score matches the two discovered key directions to the known A and B token contrasts *after fitting*. The effect fraction measures how much squared token-bit-flip response from each selected component belongs to its matched bit, including the filler contrast in the denominator. It is a function-level selectivity measure; it is not physical-head causal fidelity. The models learned A/B with supervised targets; only the reduction metric and gauge selection are answer-free.

## Interpretation and next gate

Here several inexpensive priors agree. The retrieval table has nearly two clean, axis-aligned variables already, so key, output, response, and all-pair edit sparsity select similar bases. This is evidence that the result is robust within this task family, **not** evidence that one prior is universally best. The all-pair measure even includes swaps between token roles that would not form valid natural sequences, so it should be read as a declared synthetic intervention metric. The causal-decoy toy and the quadratic-readout count toy show cases where different priors disagree.

The next discriminating experiment should introduce correlated and rare key variables with more than two useful directions, then train multiple seeds. Compare these same priors on the *same* reduced function under whole-combination holdouts, and evaluate finite component edits against the full model. The 546M model needs a separate transfer gate with fresh token swaps, many contexts, and path-output as well as task-effect validation. A weight-only reduction can be fit without examples, but selecting a human-legible gauge and claiming causal meaning still requires independent probes.

Run: `taskset -c 2-5 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/neuron_circuits/.venv/bin/python dual_joint_edit_gauge.py`
