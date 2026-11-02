# Cross-layer block subsets for the joint-language rules

The [cross-model participation study](V3_JOINT_CROSS_MODEL_RESULTS.md)
found that two trained three-block bilinear transformers solve the same
outer-bracket task but depend on different blocks when one attention
output is removed. Single-block ablations leave an ambiguity: two blocks
could be redundant, jointly necessary, or interacting nonlinearly. This
experiment evaluates all eight keep/zero combinations on fresh exact-
oracle prediction and grammar-valid edit panels, with no retraining.

[`eval_joint_block_subsets.py`](eval_joint_block_subsets.py) hooks each
removed attention block's output to zero while preserving its residual
connection and all remaining trained weights. The [raw result](joint_depth3_block_subsets_seed20261300.json)
uses the same 128-stream generator seeds for both independently trained
models. A bit string `abc` means keep blocks 0, 1, and 2 when the
corresponding bit is `1`. Native `111` teacher KL is zero by construction.

| Kept blocks | First model outer-edit L1 | Second model outer-edit L1 | First model inner-edit L1 | Second model inner-edit L1 |
|---|---:|---:|---:|---:|
| `000` | 2.000 | 2.000 | 2.000 | 2.000 |
| `001` | 0.621 | 1.898 | 2.124 | 2.053 |
| `010` | 1.999 | 0.562 | 2.024 | 2.049 |
| `011` | 1.340 | **0.046** | 1.894 | 1.512 |
| `100` | 1.978 | 2.005 | 2.095 | 2.063 |
| `101` | **0.0008** | 1.918 | 0.577 | 0.932 |
| `110` | 1.973 | **0.028** | 2.128 | 1.935 |
| `111` | 0.0004 | 0.0005 | 0.002 | 0.001 |

The first model has a particularly clean two-block outer-close path in
this intervention family: `101` keeps blocks 0 and 2 and reproduces the
outer relevant-edit effect essentially exactly, while `100` and `001`
do not. The second model needs block 1 and benefits from *either* block 0
or block 2: `110` and `011` are approximate outer-close solutions, while
`010` is much worse. This is an OR-like alternative-support pattern for
the outer rule under finite block ablations. It is not proof of two
separate native algorithms or a unique minimal arithmetic circuit.

The distinction between outer and inner closing matters. None of the
two-block subsets preserves inner opener-edit effects closely in either
model; even the first model's `101` path has inner-edit L1 **0.577**.
Other tasks also degrade under the outer-specific reductions. For the
first model's `101`, verb/object oracle KL is **0.679/0.281** versus native
**0.083/0.114**. For the second model's `110`, it is **0.654/0.394**
versus native **0.097/0.113**. Thus a compact path sufficient for one
rule is not a compact model of the whole language. The raw JSON records
all verb and object rule edits and nuisance controls separately.

These results explain why a first-layer eigenbasis could separate outer
closing in one seed but not the other. They also give a useful benchmark
for the next featurizer: recover small *task-conditional cross-layer paths*
that preserve relevant edits and nuisance invariance, then test the same
method on verb, object, and inner-close rules. A path should be assessed
against the trained model and the exact oracle; these differ substantially
for object induction. This block-subset map is a coarse causal scaffold,
not the requested atom-level interpretation. Further outer-only subset
tuning would add little; broaden task and surface coverage while testing
the featurizer across both model seeds.
