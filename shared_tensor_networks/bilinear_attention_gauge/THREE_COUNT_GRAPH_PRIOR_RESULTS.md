# A reuse-aware graph prior on learned quadratic attention

**Independent-seed follow-up:** [ten newly trained models](THREE_COUNT_GRAPH_PRIOR_FRESH_RESULTS.md)
show only partial transfer. The uniform-cut pruning benefit remains but
costs more variable alignment; the large frequency-cut benefit below does
not replicate with the hyperparameters frozen.

The [from-scratch three-count models](THREE_COUNT_SCRATCH_COMPOSITION_RESULTS.md)
expose a conflict: a sparse key basis recovers the intended A/B variables,
while a basis that makes single-axis deletions additive mixes those variables.
This experiment asks whether a **label-free** preference for a small quadratic
computation graph can supplement sparse keys without forcing that mixture.

For each of ten independently trained models, use the existing weight-only
uniform rank-two cut or unlabeled-frequency rank-two cut. Rotate only within
the fixed two-dimensional key subspace. Every rotation has the same reduced
function and projector. Let `D` be the two key directions, `C_q` map key
coordinates to readout logits for query `q`, and `Q` be the learned symmetric
quadratic readout. The induced computation is `H_q = C_qᵀ Q C_q`, represented
by three possible shared monomial nodes: `z₀²`, `z₀z₁`, `z₁²`. Group each
node's coefficient across the two query types, so one reused product is one
node rather than two occurrences. The objective is

`||D||₁ / min_rotation ||D||₁ + λ Σ_nodes min(1, ||coefficient_node|| / (τ ||H||))`.

This is a surrogate description-length cost, not a proof of minimum circuit
size. The A/B/C labels were excluded from reduction, gauge choice, and
pruning-node selection; they were used afterward to score designed-variable
alignment. We swept `τ ∈ {.03,.1,.2}` and `λ ∈ {0,.1,.3,1,3,10}` on a
1024-angle grid.

At `τ=.1, λ=.3`, the new prior retains much of key sparsity's variable
alignment while making one quadratic node cheaper to omit:

| Reduction and gauge | Matched variable cosine ↑ | Held-out one-node deletion relative MSE ↓ | Shifted-count deletion MSE ↓ |
|---|---:|---:|---:|
| Uniform, key L1 | .947 | .043 | .048 |
| Uniform, context-edit | .829 | .029 | .032 |
| Uniform, graph prior | **.937** | **.028** | **.029** |
| Frequency, key L1 | .965 | .017 | .015 |
| Frequency, context-edit | .866 | .014 | .015 |
| Frequency, graph prior | **.957** | **.001** | **.001** |

For pruning, choose the lowest-energy quadratic node on 4096 unlabeled fit
contexts, then delete that same node on 4096 independent held-out contexts
and 4096 higher-key-count contexts. The error is the centered omitted-node
variance divided by the centered variance of the complete quadratic output.
It is a function-level compression check on the quadratic channel, **not**
four-output task accuracy or a physical internal-node intervention. The
uniform and frequency cuts are distinct reduced functions; compare gauges
within a cut, not raw numbers between cuts.

The downstream task check uses the same 30,000-sentence independent test
and whole-combination holdout as the original scratch analysis. It deletes
the fit-selected node from parity while retaining the reduced A/B/C logits:

| Reduction and gauge | Full-test parity accuracy | Rare-holdout parity accuracy | Full-test four-output accuracy |
|---|---:|---:|---:|
| Uniform, unpruned rank 2 | .733 | .158 | .578 |
| Uniform, key L1 then prune | .710 | .182 | .563 |
| Uniform, graph prior then prune | **.728** | .175 | **.571** |
| Frequency, unpruned rank 2 | .661 | .734 | .523 |
| Frequency, key L1 then prune | .660 | .747 | .530 |
| Frequency, graph prior then prune | **.667** | .733 | .528 |

The graph-pruned parity is close to its unpruned rank-two function, especially
under the frequency cut. It does **not** rescue the poor rank-two four-output
accuracy. On the uniform cut, rare four-output accuracy remains .100 under
all these gauges because the three ordinary bits already fail there. This
separates economical representation of a reduced function from selecting a
reduction that preserves the actual task.

The prior has an observable cost: at `τ=.1, λ=1`, uniform-cut alignment falls
to .909 while deletion MSE improves to .019. At small `λ=.1`, it behaves
almost like pure key sparsity. The `λ=.3` point above was identified within
this sweep, so it needs selection on separate models before making a
generalization claim. This experiment shows a useful tradeoff *on these
trained toys*, but does not identify the physical computation graph:
functionally equivalent circuits remain possible.

Next: select `τ,λ` using development toy seeds, test on wholly unseen toy
mechanisms and architectures, and score interventions. Then transfer the same fixed objective to the first-layer
all-head 546M cut, where no A/B/C oracle is available and full-context
effects determine whether the components are useful.

Reproduce with `three_count_graph_prior.py` and
`three_count_graph_prior_task.py`; per-seed values and the full sweep are in
`three_count_graph_prior.json` and `three_count_graph_prior_task.json`.
