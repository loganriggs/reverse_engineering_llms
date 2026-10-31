# Does task data make readable product nodes physically realizable?

The [coefficient-space comparison](SHARED_PAIR_PHYSICAL_NODE_ALIGNMENT_RESULTS.md) found that readable nodes in a rank-three, weight-only canonicalization are generally outside the span of the trained models' six physical attention-head-pair terms. That proves a lack of exact polynomial identity, but task sentences occupy a restricted set of token-count vectors. This follow-up asks whether the *effects of deleting those nodes* can nevertheless be reproduced on that distribution.

For each of ten paired-token and ten dense-token three-head models, we compare the three strongest semantic product nodes in two gauges: ordinary SVD and the readable joint-nullspace gauge. The reduction is always the same weight-only uniform rank-three function. No answer labels or task outcomes enter the fits. For each model, 20,000 independently generated **seen-combination** sentences fit physical edits; a separate 30,000-sentence test set includes the rare whole-combination holdout. We use four comparators:

1. The best single physical head-pair term with a fitted scalar.
2. A feasible edit that attenuates each real head output by a fixed gain in [0,1] before the quadratic readout. It has three fitted gains and induces the six pair coefficients `1 - gain_i * gain_j`.
3. An unrestricted linear combination of all six physical head-pair terms, fitted on seen combinations. This is a generous function-space *upper bound* for these particular physical terms; arbitrary coefficients may not be executable by head attenuation.
4. The same unrestricted fit allowed to see an independent full-distribution sample, including the rare combination. This is an oracle diagnostic, not a valid holdout method.

Mean test MSE divided by the mean squared semantic node effect, over the three selected nodes and ten seeds per coding:

| Coding, gauge | Single pair | Physical three-head attenuation | Six-pair span, seen fit | Six-pair span, full-distribution oracle |
|---|---:|---:|---:|---:|
| Paired, SVD | .694 | .613 | .457 | .457 |
| Paired, joint nullspace | .742 | .594 | .518 | .518 |
| Dense, SVD | .692 | .615 | .456 | .455 |
| Dense, joint nullspace | .764 | .600 | .525 | .524 |

The data-aware six-pair fit captures some task-distribution structure, but leaves about half the semantic effect energy unexplained on new sentences. Even the oracle fit barely improves ordinary-distribution generalization. This is not merely a shortage of fitting examples or missing rare-combination examples. The executable attenuation edit is worse than the permissive span, as expected.

The mean hides a useful distinction. In both codings, joint-nullspace A–B deletion has roughly **.90–.94 relative test MSE** under the permissive six-pair fit, whereas B–C and A–C each have roughly **.31–.33**. On the rare combination, A–B's residual MSE is about **5.7–6.0 times** its full-test target energy. The rare subset's own target energy can be nearly zero for some other nodes, so the JSON reports both its own-denominator error and this stable full-test-energy normalization. Across nullspace nodes, physical six-pair deletion agrees with the semantic-deletion parity signs on only about .68 of full test sentences; this includes many sentences whose decision never flips. On sentences where the semantic deletion flips at least one parity decision, exact two-output sign agreement averages only .10–.12. That last measure is stringent and conditioned on semantic flips; it is descriptive rather than a calibrated behavioral causal score.

The result narrows the interpretation: the sparse, reusable product graph is a valid **external decomposition of the native function**, including rare-task behavior, but its node deletions are not generally equivalent to attenuating trained heads. A distribution-weighted metric helps two products, but does not rescue all three or identify the physical graph. This does not rule out a different physical partition or context-dependent internal edit. The next physical-circuit test should expand the candidate edits to QK branches and within-head subspaces, fit on seen combinations, and reserve whole combinations and new edit strengths for evaluation. If those also fail, keep the readable product graph explicitly at function level.

Reproduce with `shared_pair_data_equivalent_node_edits.py`; the JSON records each seed, gauge, node, physical coefficients, fitted head gains, test-stratum errors, and parity-sign diagnostics. CPU only; no new model training.
