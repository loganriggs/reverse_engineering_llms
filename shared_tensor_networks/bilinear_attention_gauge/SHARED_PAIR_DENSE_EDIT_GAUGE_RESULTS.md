# Unlabeled token-swap sparsity recovers a trained dense-code function

The [dense-token shared-product task](DENSE_CODED_SHARED_PAIR_RESULTS.md)
is a trained control: eight key tokens each encode all three binary
attributes, so token-entry L1 fails to recover individual variables.
Ten independently trained three-head bilinear-attention models reach
100% five-output accuracy, including a whole-combination holdout.
Their exact first-layer table and learned quadratic readouts are known.
Both uniform-token and unlabeled-frequency rank-three reductions
preserve the unpruned task in every seed.

We fit an exact **orthogonal gauge** to each fixed rank-three function.
For each of the 28 unordered key-token pairs, take the difference of
their reduced key-coordinate vectors and minimize the mean L1 norm of
these differences. This uses a declared, **uniform all-pair input-edit
measure**. It does not use task labels, token attribute labels, the
quadratic readout, or sampled sentences. Hamming-one swaps are a
structure-informed positive control. Every rotation leaves the fixed
projector and unpruned token function unchanged; maximum checked
relative table discrepancy across saved gauges is `1.4e-15`.

| Rank-three cut / gauge | Known-variable cosine | One dominant quadratic node per output/query cell | Five-output accuracy after three-node pruning: full / rare holdout |
|---|---:|---:|---:|
| Uniform / SVD | .907 | .647 | .828 / .763 |
| Uniform / token-entry L1 | .700 | .393 | .688 / .632 |
| Uniform / all 28 token swaps | **.99997** | .992 | **1.000 / 1.000** |
| Uniform / Hamming-one swaps | **.99997** | .992 | **1.000 / 1.000** |
| Frequency / all 28 token swaps | **.99997** | .992 | **1.000 / 1.000** |
| Frequency / Hamming-one swaps | **.99996** | .992 | **1.000 / 1.000** |

Numbers are means over ten independently trained models. For the
uniform cut, the all-pair method's minimum variable cosine is .99994;
minimum post-pruning full and rare accuracy is 1.000. The three nodes
are selected from the learned quadratic readout by **global coefficient
energy**, without task labels. Designed-variable alignment and task
accuracy are measured only after gauge fitting and node selection.
This improves substantially over token-entry sparsity on this coding
without needing hand-supplied Hamming-one edges.

This is still not blind discovery from arbitrary text. The eight-token
input code has a complete three-bit cube structure, which the uniform
pair ensemble exposes. The rank-three cut and three-node budget are
supplied by prior task-faithfulness analysis, and the quadratic readout
was trained with supervised tasks. The earlier
[physical-branch red team](SHARED_PAIR_BRANCH_CAUSAL_RESULTS.md) shows
that this ordinary rank-three function space badly misses separate
QK1/QK2 edits; no exact rotation can repair those missing directions.
The result concerns a reusable **function-level** graph, not the
physical arithmetic circuit.

## Coverage, not just draw count

We fitted the edit-L1 gauge from random subsets of the 28 swaps on
three trained models. Thirty panels per model and sample size give 90
fits per row. Success means post-fit cosine above .98 with known
variables. The fraction whose quadratic cells have single-node energy
above .95 is almost identical. These coverage runs do **not** repeat
the 30,000-sentence task-pruning evaluation.

| Draw scheme | Draws | Mean distinct swaps | Basis successes / 90 |
|---|---:|---:|---:|
| With replacement | 8 | 7.0 | 25 |
| With replacement | 28 | 17.7 | 41 |
| With replacement | 56 | 24.0 | 61 |
| Without replacement | 8 | 8 | 33 |
| Without replacement | 16 | 16 | 53 |
| Without replacement | 20 | 20 | 63 |
| Without replacement | 24 | 24 | **89** |
| Without replacement | 28 | 28 | **90** |

Covering nearly every edit direction is feasible in this tiny code.
Repeatedly sampling an imbalanced set can shift the L1 optimum even
when its distinct-pair count matches a more balanced panel. The
hypothesis for larger vocabularies is that diverse intervention support
and controlled weighting can matter more than raw edit count. That
requires tests with larger trained languages, noise, and unseen edits.
The 90 fits per row are repeated panels over **three models**, not 90
model replications.

Reproduce with `shared_pair_dense_edit_gauge.py` and
`shared_pair_dense_edit_coverage.py` under both schemes. The complete
gauges, task scores, sampled swap lists, and coverage results are in
their corresponding JSON files.
