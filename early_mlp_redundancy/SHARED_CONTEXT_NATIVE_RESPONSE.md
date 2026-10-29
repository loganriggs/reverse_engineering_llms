# M53: local direction accuracy and native causal fidelity rank differently

**The token-mean advantage in local geometry does not uniformly survive the downstream reader.** On natural text, learned shared components favor token means (response error 0.2673 versus donor 0.3083), while random shared components favor donors (0.3875 versus token mean 0.4466). Zero response scores 1. Token means beat donors in only 5/8 individual natural shared-component models; exclusive and union comparisons are 4/8 and 3/8. There is no universal winning predictor.

## Native causal experiment

All 192 original inputs, eight frozen shared models, and exclusive/shared/union components are retained. Only positions eligible under the activation-blind M50 same-token/group donor inventory are intervened on. The teacher subtracts the recipient's component from its original native MLP1 write. Each predictor subtracts its proposed direction with oracle recipient magnitude. Down bias is preserved and downstream layers run normally. Zero and missing directions remain in the population. Token means and unconditional means use only natural training documents 0–31; no task evaluation fitting.

For each output position, measure the change in log probabilities caused by the recipient intervention and by the predictor intervention. Weight squared prediction errors by the original native probabilities, sum over vocabulary and positions, and divide pooled error by pooled teacher-response energy. Zero response therefore scores 1 where energy is nonzero. This is response prediction, not standalone disruption size, task accuracy, or output KL.

## Results

Pool energies across two seeds and two source objectives within each grouping. Candidate models define different targets; these tables do not rank architectures on a common target.

| Cohort | Grouping | Component | Donor | Group mean | Token mean | Scrambled |
|---|---|---|---:|---:|---:|---:|
| task/train | random | exclusive | 0.5675 | 1.1970 | 0.9016 | 1.5702 |
| task/train | random | shared | 0.2005 | 0.9511 | 0.4243 | 1.2173 |
| task/train | random | union | 0.2294 | 1.0571 | 0.6045 | 1.2388 |
| task/train | coactivation | exclusive | 0.4336 | 1.2206 | 0.8798 | 1.4920 |
| task/train | coactivation | shared | 0.2044 | 0.6867 | 0.2589 | 1.1734 |
| task/train | coactivation | union | 0.1973 | 0.8236 | 0.4416 | 1.2870 |
| task/evaluation | random | exclusive | 0.6057 | 1.9103 | 0.5537 | 1.6385 |
| task/evaluation | random | shared | 0.6083 | 0.7161 | 0.1695 | 1.5246 |
| task/evaluation | random | union | 0.4063 | 0.8067 | 0.2121 | 1.4188 |
| task/evaluation | coactivation | exclusive | 0.4227 | 0.8906 | 1.9528 | 1.6187 |
| task/evaluation | coactivation | shared | 0.1588 | 0.6082 | 0.2427 | 1.2355 |
| task/evaluation | coactivation | union | 0.2304 | 0.6062 | 0.2974 | 1.4279 |
| natural/evaluation | random | exclusive | 0.7918 | 1.2902 | 0.8719 | 1.2370 |
| natural/evaluation | random | shared | 0.3875 | 0.7198 | 0.4466 | 1.1131 |
| natural/evaluation | random | union | 0.4306 | 0.8460 | 0.5057 | 1.1036 |
| natural/evaluation | coactivation | exclusive | 0.7020 | 1.1162 | 0.7789 | 1.3401 |
| natural/evaluation | coactivation | shared | 0.3083 | 0.5997 | 0.2673 | 1.1728 |
| natural/evaluation | coactivation | union | 0.3261 | 0.6037 | 0.2766 | 1.1665 |

The strongest caveat is the learned exclusive component on task evaluation: token-mean response error is 1.9528, worse than zero response, although its local direction error was 0.7038. Local vector accuracy can hide mistakes in directions that matter disproportionately to the downstream model. Conversely, shared token means and donors both predict useful fractions of native response energy in the pooled cohorts.

## What this says about computation

- **Function geometry:** token-conditioned means describe a substantial part of shared directions. This is compatible with the earlier token-table evidence, but oracle magnitude and support still carry recipient information.
- **Implementation:** a token-conditioned direction can be distributed across channels and shared atoms. Architectural gate protection does not establish that native neurons are redundant.
- **Downstream use:** the reader weights directions unevenly. Better Euclidean reconstruction need not give a better causal surrogate. The rank reversals establish this empirically; they do not alone distinguish reader anisotropy from finite-intervention nonlinearities or domain shift.
- **Redundancy:** none of these external featurizer substitutions proves that many native parts can be removed with the same computation preserved. Earlier native removal/compensation experiments remain the evidence for that separate question.

The next causal-supervision objective should predict native responses on a common intervention set, preserving comparisons across featurizers. It should not merely reward smooth directions or large interventions. These results do not validate semantic manifolds or new-task discovery.

## Verification

8967 native forward passes produced 23040 predictor records for 4608 target interventions, including 2781 zero-target interventions. The producer checks cached native writes against every original input, component additivity before normalization, norm matching and zero handling. Eighty fixed task/natural shared-component cases independently recompute full-vocabulary, all-position logsoftmax response metrics with NumPy; maximum absolute discrepancy is 1.11e-16. A separate audit verifies complete inventory, protocol/source hashes, zero identities, forward accounting, and all 360 summary aggregations. It does not independently rerun all model forwards.

## Stop and return to the broader loop

This closes the conditional-direction control series M50–M53. Do not add donor-seed, amplitude, smoothing, count-threshold, or membership searches to improve these numbers. The deliverable is a concrete failure mode for reconstruction-based featurizer validation and a frozen native-response benchmark.

Next reconcile these findings with the existing common-target scalar/block/shared causal comparisons and the preserved task-family split. Produce a joint evidence table separating reconstruction, native-response fidelity, task prediction, and genuinely unseen task transfer. Use that audit to specify one broader task-coverage experiment with training-only selection and whole-family holdouts; do not treat a new score on these familiar 192 inputs as new-task discovery. Before further training, identify which existing supervised targets actually help causal fidelity and what missing task diversity the new experiment supplies.
