# Reduction metric × legibility prior on trained two-head retrieval

The two-output retrieval toy is a useful transfer test because both trained attention heads are needed to solve the task. For each of ten independently trained checkpoints, we constructed the exact *combined* token-function table (four query/output rows by six key tokens), reduced it to rank two, and selected an orthogonal basis within that reduced function. The basis change preserves the rank-two function exactly; it does not rotate the physical heads or certify a unique circuit.

## Factorial design

The reduction metric was either uniform over all 24 token-function cells (weight-only, no sampled inputs), estimated from 100,000 unlabeled training-distribution sequences, or a 50:50 mixture. The frequency metric uses query/key occurrence counts, with no output labels. For each reduced function, we scanned 4,096 rotations and compared the SVD baseline with eight objectives: key-dictionary L1, output-row L1, normalized sum of both factor L1s, component-response L1, whole-sequence component-response L1 under uniform or correlated sequence laws, mean L1 change across all 15 unordered key-token swaps, and mean L1 change across three *known-role* bit flips. The all-pair objective uses no role or task labels; the known-role objective is a positive control. The correlated sequence law is a declared **synthetic stress test** with P(A=B)=.9, not the models' training distribution.

| Reduction | Gauge | Mean A/B alignment | Worst seed | Intended bit-effect fraction | Full / rare task accuracy |
|---|---|---:|---:|---:|---:|
| Uniform | SVD | .884 | .737 | .829 | 1.00 / 1.00 |
| Uniform | key L1 | .975 | .929 | .997 | 1.00 / 1.00 |
| Uniform | row L1 | .976 | .931 | .999 | 1.00 / 1.00 |
| Uniform | response L1 | .976 | .931 | .999 | 1.00 / 1.00 |
| Uniform | sequence-response L1 | .731 | .678 | .569 | 1.00 / 1.00 |
| Uniform | correlated sequence-response L1 | .730 | .678 | .568 | 1.00 / 1.00 |
| Uniform | all-pair edit L1 | .976 | .930 | .999 | 1.00 / 1.00 |
| Uniform | known-role edit L1 | .975 | .929 | .999 | 1.00 / 1.00 |
| Empirical | SVD | .886 | .737 | .829 | 1.00 / 1.00 |
| Empirical | correlated sequence-response L1 | .701 | .679 | .514 | 1.00 / 1.00 |
| Empirical | all-pair edit L1 | .978 | .953 | .999 | 1.00 / 1.00 |
| Half mixture | SVD | .886 | .738 | .830 | 1.00 / 1.00 |
| Half mixture | correlated sequence-response L1 | .704 | .680 | .520 | 1.00 / 1.00 |
| Half mixture | all-pair edit L1 | .977 | .945 | .999 | 1.00 / 1.00 |

The normalized two-factor L1 condition is similar: alignment .975/.978/.977 for uniform/empirical/mixed. The remaining conditions and per-seed results are in `dual_joint_edit_gauge.json`. All nine gauges preserve the selected rank-two table to numerical precision. All 30 rank-two reductions retain 100% joint task accuracy on the exhaustive test and on the rare-combination subset. Mean relative uniform table MSE is .00491 for uniform, .00737 for empirical, and .00559 for mixed.

The alignment score matches the two discovered key directions to the known A and B token contrasts *after fitting*. The effect fraction measures how much squared token-bit-flip response from each selected component belongs to its matched bit, including the filler contrast in the denominator. It is a function-level selectivity measure; it is not physical-head causal fidelity. The models learned A/B with supervised targets; only the reduction metric and gauge selection are answer-free.

## Interpretation and next gate

The key distinction is the **level where response sparsity is measured**. A complete token-function component has one key token at a time; sparsifying its entries or individual pair responses favors the two separate A/B variables. Whole-sequence contributions sum over the three keys before applying L1. A and B then coactivate, and a rotated sum/difference basis lowers L1 by cancellation: the simple vector (1,1) has L1 norm 2, while its 45-degree rotation is (√2,0), with L1 norm √2. Under the uniform reduction, the whole-sequence prior reduces its own mean L1 from 22.59 for the all-pair-edit basis to 21.42, but lowers variable alignment from .976 to .731 and intended-bit effect concentration from .999 to .569. The correlated stress law pushes empirical-reduction alignment down to .701 and effect concentration to .514. The full predictions are unchanged because these are exact rotations of the same reduced function.

Component deletion also becomes less selective. For uniform reduction, deleting the A component selected by all-pair edit L1 leaves B accuracy at 1.00 and takes A accuracy to .50; deleting its B component does the reverse. Under uniform whole-sequence response L1, deleting the matched A component leaves B accuracy only .844, and deleting the matched B component leaves A accuracy .794. These are task-level consequences of the **reduced function's** component deletion, not interventions on physical attention heads. The all-pair measure includes role-invalid swaps, so it should be read as a declared synthetic intervention metric. This result matches the earlier causal-decoy and quadratic-readout controls: sparsity can reward mixtures when variables coactivate.

The next gate should train the model itself on a correlated and rare-variable sequence law with more than two useful directions, then test whether this prior conflict persists and whether all-pair edit selection transfers to held-out edit families. This run varied the gauge-evaluation law, not the model-training law. Finite component edits should also be compared against the full model. The 546M model needs a separate transfer gate with fresh token swaps, many contexts, and path-output as well as task-effect validation. A weight-only reduction can be fit without examples, but selecting a human-legible gauge and claiming causal meaning still requires independent probes.

Run: `taskset -c 2-5 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/neuron_circuits/.venv/bin/python dual_joint_edit_gauge.py`
