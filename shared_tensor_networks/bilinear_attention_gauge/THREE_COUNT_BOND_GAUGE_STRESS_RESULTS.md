# An exact bond gauge can hide a task-critical direction from local energy

The [trained-toy rank test](THREE_COUNT_RANK_SELECTION_RESULTS.md) selected
rank three reliably when all three task variables had comparable scale. We
now stress the **canonicalization-before-truncation** step without changing
the complete model function. This is a controlled mathematical
reparameterization of each trained model's exact `6×8` token function; it
does **not** claim the physical attention weights use the constructed gauge.

For each of ten independently trained three-head toy-language models, take
the exact combined function table `T=UΣVᵀ` and write its six-dimensional
factorization `T=AB`, with `A=UΣ^(1/2)` and `B=Σ^(1/2)Vᵀ`. Apply the exact
internal-bond gauge

`A' = A diag(1,1,1/ε,1,1,1)`,
`B' = diag(1,1,ε,1,1,1) B`.

Then `A'B'=T` to float64 precision for every `ε`: all original task logits,
rare-combination behavior, and finite token-edit responses are unchanged
*before truncation*. We compared (a) truncating `B'` by its own unweighted
SVD and selecting the smallest rank meeting an internal-factor error
threshold, (b) selecting rank from answer-free **decoded** within-pair
token-edit response error after `A'`, and (c) SVD truncation of the full
function `T`. The latter fixes this bond gauge for a declared Euclidean
input/output metric. All three rank criteria use no task labels; labels
enter only in the subsequent test on 30,000 independent sentences per seed.

At a 5% relative error threshold:

| ε | Internal-factor energy: ranks over ten seeds; rare held-out accuracy | Decoded edit response: ranks; rare accuracy | Full-function SVD: ranks; rare accuracy |
|---:|---|---|---|
| 1 | 3 in all; 100% | 3 in all; 100% | 3 in all; 100% |
| .3 | 3 in all; 100% | 3 in all; 100% | 3 in all; 100% |
| .1 | **2 in all; 10%** | 4–6; 100% | **3 in all; 100%** |
| .03 | **2 in all; 10%** | 6 in all; 100% | **3 in all; 100%** |

At `ε=.1`, the rank-two cut of the internal factor has only **2.9%** mean
relative error in `B'`, yet after applying `A'` its full-function error is
**31.5%** and its worst-query token-edit error is **32.6%**. At `ε=.03`,
the same quantities are 2.5%, 31.5%, and 32.6%. The full-function rank-three
cut instead has mean relative table error `.00041` and 100% rare task
accuracy. Even a stricter 1% internal-factor threshold can select the wrong
directions: at `ε=.03` it chooses rank three or four, but rare accuracy still
averages 10%; the reduced factor has used those positions on higher-energy
irrelevant directions before reaching the shrunken third coordinate.

This is an exact, small counterexample to interpreting *local factor
energy* as functional importance before specifying and fixing the bond
metric. Measuring decoded edit fidelity rescues behavior but may need all
six coordinates in a poor gauge. Canonicalizing the **complete function**
under the declared metric makes rank three sufficient across all tested
gauges. This is the practical value of a canonical reduction step: it can
remove a large class of representation-dependent truncation mistakes
before applying a sparsity or grouping prior.

The full-function SVD is not a unique semantic decomposition: its axes
depend on the input/output metric, and near-degenerate singular subspaces
can rotate. The construction is a matrix factorization of a toy token
function, not a closed-loop attention-network canonical form. It motivates
testing environment-aware bond metrics on larger implicit attention
networks, then assessing whether the resulting reduced subspace supports
stable, legible coordinates and held-out interventions.

Reproduce with `three_count_bond_gauge_stress.py`; all per-seed rank cuts,
errors, task scores, and both threshold sweeps are in
`three_count_bond_gauge_stress.json`.
