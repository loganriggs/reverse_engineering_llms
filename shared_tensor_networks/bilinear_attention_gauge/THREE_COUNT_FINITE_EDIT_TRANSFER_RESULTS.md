# Finite token edits in the trained model versus gauge-attributed reductions

The [edit-family holdout](THREE_COUNT_EDIT_FAMILY_HOLDOUT_RESULTS.md) found a rank-three gauge using cross-pair token swaps while excluding all four direct within-pair bit flips. We now apply those withheld substitutions to the **trained model itself**, not just to the token table used for fitting. Ten from-scratch trained, three-head bilinear-attention models each receive independent ordinary and higher-key-count contexts. Edits include both directions of A, B, C, and filler token flips, stratified into common query, rare seen query, and the rare whole-combination holdout. There are 51,200 paired before/after token interventions in total.

The exact combined token table plus learned quadratic readout reproduces direct full-model logits before and after every edit to maximum absolute error `2.1e-12`. The rank-three uniform weight-only cut and unlabeled-frequency cut are frozen before gauge choice. The gauge fit excludes the direct bit-flip edge family and all new contexts. Known A/B/C labels are used only afterward to match components for scoring.

## Finite edit fidelity and attribution

For each finite edit, the table reports mean squared error of the *four-logit change* relative to the full trained model's change. The component columns use a quadratic Shapley decomposition of the reduced function's finite change: if `Δb = Σc Δb_c` is the change in its three bit logits and `p(b)=bᵀQb`, component `c` receives `2bᵀQΔb_c + Δb_cᵀQΔb` of the parity change. The component contributions sum exactly to the reduced function's total edit effect. Metrics macro-average the A/B/C families, three query strata, and ten seeds; filler edits are a separate negative-control family in the JSON.

| Reduction and context | Whole reduced edit vs full model MSE ↓ | SVD matched-component MSE ↓ | Key-L1 matched-component MSE ↓ | Gauge fitted without bit flips: matched-component MSE ↓ | That gauge: matched-component energy fraction ↑ |
|---|---:|---:|---:|---:|---:|
| Uniform; ordinary | .0004 | 3.226 | .0020 | **.0009** | .9996 |
| Uniform; higher key count | .0005 | 3.796 | .0023 | **.0011** | .9995 |
| Frequency; ordinary | .0009 | 3.101 | .0026 | **.0013** | .9996 |
| Frequency; higher key count | .0011 | 3.629 | .0031 | **.0015** | .9996 |

The SVD gauge's matched component contains only about .69–.72 of the component-attributed edit energy and is a poor standalone prediction of the full finite effect. The gauge fitted on the other edit edges captures at least .9995 on average and predicts the full edit effect nearly as well as the entire rank-three surrogate. Key-entry L1 also works well on this paired-token code, though less precisely. The all-28-edge gauge is slightly better again, but it directly trains on the tested edit family and is therefore not the transfer result.

The reduction measure still matters independently of the gauge. On ordinary contexts with the common query, the empirical rank-three cut has mean relative finite-edit error about `2e-5`, versus `2.9e-4` for uniform. On the rare whole-combination query, the ordering reverses: empirical is about `.00166` versus uniform `.00056`. The same pattern survives the higher-count shift (`2e-5` versus `.00037` on common contexts; `.00197` versus `.00067` on the rare holdout). Both cuts are task-accurate, yet weighting by training usage reallocates fidelity away from rare edits.

## What this establishes

These are **finite input interventions in the trained model**, followed by comparison with a reduced function and its coordinate attributions. They show that a label-free edit prior can yield a compact description whose individual components predict held-out finite effects across contexts and a count shift. They do **not** show that the components coincide with physical QK branches, heads, or independently patchable neurons. The first-layer lookup table makes the intervention function exactly enumerable; a later layer with context-dependent keys and queries is a stronger test. The next 546M-model gate should use unseen token-edit families and many independent contexts, and should compare path-output fidelity, selective full-model effects, and ordinary task behavior separately.

Reproduce with `taskset -c 2-5 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/neuron_circuits/.venv/bin/python three_count_finite_edit_transfer.py`. Event-level strata, per-output errors, and component fractions are in `three_count_finite_edit_transfer.json`.
