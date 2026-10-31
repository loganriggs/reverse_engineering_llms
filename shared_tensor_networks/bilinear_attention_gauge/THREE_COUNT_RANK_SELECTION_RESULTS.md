# Answer-free edit fidelity selects the required rank in a trained toy

The [bracket head control](BRACE_COMPETENCE_AND_GROUP_TRANSFER_RESULTS.md)
showed that a low-energy second edit direction mattered on unseen token
identities. To check whether edit fidelity can choose group size without
answers, we reused ten independently trained three-head bilinear-attention
toy-language models. Each model solves three count-sign bits and a learned
quadratic parity output. A complete rare A=0/B=1/C=0 query combination was
excluded from supervised training and validation.

We constructed each model's exact `6×8` combined query/output-by-key-token
table and fitted rank-one through rank-six functions with three measures:
uniform table cells (weight-only SVD), unlabeled training pair frequency,
and a half-uniform mixture. For each cut we computed **answer-free** squared
error of the three-output finite token-edit response, stratified by common
and rare query. The selection panel uses four within-pair token swaps; a
separate 24-swap cross-pair panel audits the choice. We selected the smallest
rank whose *worst-query* within-pair relative edit MSE was below a declared
threshold. Task labels, the parity rule, and held-out-combination outcomes
were used only after rank selection.

| Cut | Rank | Mean within-pair error, common / rare | Mean cross-pair error, common / rare | Full / rare held-out all-four accuracy |
|---|---:|---:|---:|---:|
| Uniform | 2 | .325 / .300 | .326 / .301 | .578 / .100 |
| Uniform | 3 | .00001 / .00002 | .00036 / .00071 | 1.000 / 1.000 |
| Usage-weighted | 2 | .313 / .338 | .313 / .339 | .523 / .544 |
| Usage-weighted | 3 | <.00001 / .00005 | .00002 / .00173 | 1.000 / 1.000 |
| Half-uniform | 2 | .318 / .317 | .318 / .317 | .507 / .187 |
| Half-uniform | 3 | <.00001 / .00002 | .00013 / .00119 | 1.000 / 1.000 |

Every threshold in `{.001, .01, .05}` selects **rank three in all ten
seeds and all three measures**, and all 30 selected cuts retain 100% task
accuracy on an independent 30,000-sentence test set and the withheld rare
subset. At rank two, the uniform cut has lower broad table error than the
usage-weighted cut (`.315` versus `.324` relative MSE), but much lower rare
task accuracy (`.100` versus `.544`). This is another example of task score
and function fidelity ordering differently. At rank three, usage weighting
retains common cross-pair edits more closely but has higher rare-query
cross-pair error than uniform weighting (`.00173` versus `.00071`), despite
both cuts being fully task-accurate.

This is a **positive control for rank selection**, not proof of new-rule
discovery or a solution to the low-energy-axis problem. The exact token
table has only six nonzero singular directions and its three task variables
are comparably strong; rank two makes a large, easily detected edit error.
Moreover, all eight token cells appear in both reduction objectives; the
selection swaps are a separate metric but not unseen tokens. The next toy
needs a trained competent *weak or rare* rule competing with a strong
nuisance response, and rank choice must be tested on withheld rule
combinations and perturbation types. A useful stress test is whether an
answer-free edit criterion retains that weak direction when an ordinary
energy cutoff discards it.

Reproduce with `three_count_rank_selection.py`. The per-seed, per-measure,
per-rank errors, task scores, and selections are in
`three_count_rank_selection.json`. As a correctness check, the exact table
and quadratic readout reproduce native model logits on 256 independently
generated examples to maximum absolute error `4.0e-13` in float64.
