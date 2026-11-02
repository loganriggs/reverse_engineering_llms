# Sparse depth versus sparse projection factors: first language pilot

We tested whether a larger, sparser bilinear transformer reproduces a trained
two-layer language model more faithfully or exposes a simpler causal circuit.
This is a **one-seed pilot**, not an architecture ranking. The teacher is the
317-token, width-64, four-head, two-layer varied-language checkpoint in
[`surface_mixture_v2_varied_soft_models`](surface_mixture_v2_varied_soft_models/).
Its language has known lexical, cue, previous-place, and induction rules.

## Architectures and checks

[`distill_sparse_depth_vs_factor.py`](distill_sparse_depth_vs_factor.py)
constructs four relevant students:

| Arm | Construction | Initial function |
|---|---|---|
| Flat | Original 2-layer architecture | Exact teacher |
| Factor | Each attention projection becomes a product of two 64×64 matrices | Matches teacher logits to max error 1e-5 on a probe |
| Depth identity | 4 layers; teacher blocks in slots 0 and 2, added blocks initially zero-output | Exact teacher |
| Depth split | 4 layers; each teacher block copied into two successive blocks, each with half its output weight | Approximate; 120-step no-L1 distillation warmup |

Factor and both depth arms have 98,304 raw attention weights, exactly twice
the flat model's 49,152. The embeddings, LM head, and normalizations are
trainable but **not pruned**. All students are trainable; the first exploratory
run had a copied-parameter trainability mismatch and its outputs are excluded
from the comparison below. The depth-split warmup uses a separate stream so
that subsequent pruning stages see the same training stream across arms.

The matched run prunes to **absolute attention-edge counts** of 29,491,
14,746, and 6,881. Before each prune, it uses 80 AdamW steps of teacher KL
plus L1 on active attention weights; after each prune, it uses 80 steps of
teacher KL with **no L1** while enforcing hard masks. Learning rate is 5e-4,
L1 coefficient 1e-5, batch 16, sequence length 72. Teacher fidelity and
exact-oracle KL are evaluated on three independent 256-stream panels:
ordinary training distribution, increased non-immediate subject returns,
and changed determiner/parenthetical rates. Training uses CPU.

## Fidelity at matched attention-edge budget

Values below are mean per-token KL from the teacher on the ordinary panel;
smaller is better. Distillation run seed 0. The deeper models have 128 more
unpruned RMSNorm scalars than the two-layer models, so their **total** active
parameter count is 128 higher at a matched attention-edge count.

| Attention edges | Flat | Factor | Depth identity | Depth split |
|---:|---:|---:|---:|---:|
| 29,491 | 0.00153 | 0.00762 | 0.00346 | 0.05534 |
| 14,746 | 0.02465 | 0.03618 | 0.01954 | 0.13364 |
| 6,881 | 0.11014 | 0.12897 | 0.12173 | 0.24994 |

The identity-depth arm has zero output norm in both added layers after pruning.
At 14,746 edges they contain no active attention weights at all. Its score is
therefore evidence about a **pruning trajectory through a wider parameterization**,
not evidence that additional computational stages help. The depth-split arm
keeps all four blocks active, making it the relevant test of extra stages.

The factor arm improves median physical row fan-in at 29,491 edges from
39 (flat) to 19 connections. This is a local graph statistic, not yet evidence
that the intermediate coordinates correspond to readable or reusable concepts.

## Causal rule-edit test

[`eval_sparse_architecture_causal_edits.py`](eval_sparse_architecture_causal_edits.py)
uses 128 fresh generated streams and valid edits to specific rule sources.
The metric is L1 error between the model's change in next-token probabilities
and the exact generator's change; smaller is better. For 29,491 edges:

| Arm | Lexical | Induction | Previous place | Cue | Nuisance determiner |
|---|---:|---:|---:|---:|---:|
| Teacher | 0.286 | 0.274 | 0.230 | 0.228 | 0.013 |
| Flat | 0.309 | 0.281 | 0.245 | 0.236 | 0.015 |
| Factor | 0.379 | 0.319 | 0.307 | 0.318 | 0.027 |
| Depth identity | 0.322 | 0.290 | 0.262 | 0.256 | 0.019 |
| Depth split | 0.597 | 0.480 | 0.495 | 0.440 | 0.069 |

The factor model's lower local fan-in does not translate to better causal
rule-edit fidelity in this pilot. Aggregate KL, edit fidelity, and circuit
legibility must stay separate. The prior iterated L1/prune run retaining 14.23%
of the original teacher's attention weights used many more gradual rounds and
exact-oracle supervision; it is not directly comparable to the abrupt stages
here. Its result shows this pilot's low-budget failures are not a hard
expressivity limit.

### Longer no-L1 recovery

We continued the **fixed 29,491-edge masks** for another 300 distillation
steps with no L1, using the same fresh training stream for each arm. This
tests whether the initial 80 recovery steps were too short. The ordinary-panel
teacher KL fell from 0.00153 to **0.00087** (flat), 0.00762 to **0.00312**
(factor), and 0.05534 to **0.01236** (depth split). Factorization and active
depth both benefit substantially from longer recovery, but neither catches
the flat arm at this budget.

On the same 128-stream causal edit panel, lexical/induction/previous-place
effect-L1 errors after recovery are **0.301/0.279/0.244** (flat),
**0.303/0.303/0.256** (factor), and **0.392/0.360/0.345** (depth split).
The factor arm nearly closes the lexical gap, though the other rules and
nuisance-determiner edit remain worse. Recovery outputs are in
[`sparse_architecture_recovery_pilot`](sparse_architecture_recovery_pilot/).

## Interpretation and next decision

This run is a baseline for the user's deeper-sparse-distillation proposal,
not a faithful implementation of [Sparse Weight Decomposition](https://arxiv.org/abs/2608.03913)
or the alternating sparse-factor optimization in
[Double Sparse Factorization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cae31fa913f140b3126036cc474fd461-Abstract-Conference.html).
It uses balanced SVD factors followed by ordinary magnitude pruning. The
finding that this factor arm loses more teacher fidelity under our schedule
does not contradict those methods. [Gao et al.](https://proceedings.mlr.press/v306/gao26w.html)
also emphasize sparse **circuits** and interventions, which a count of zero
weights alone does not establish.

The next useful test is a smaller, better-optimized frontier around 14k–30k
edges: several seeds; gradual pruning; an explicit
sparse-factor optimizer; and a depth curriculum that keeps added blocks active
without forcing an inaccurate half-block split. A positive result must hold on
the independent surface panels **and** the causal edits, then show sparse
source-to-target paths or reusable intermediate units. Stop the depth arm if
the extra blocks remain functionally zero or if it loses causal fidelity at
matched edge count after comparable optimization. Training on diverse task
families and whole-task holdouts remains the broader research loop.

Raw matched outputs are in
[`sparse_architecture_matched_pilot`](sparse_architecture_matched_pilot/).

To reproduce from the repository root with the local PyTorch environment:

```bash
/home/loganriggs/Coding/neuron_circuits/.venv/bin/python \
  language_task_suite/distill_sparse_depth_vs_factor.py \
  --output-dir language_task_suite/sparse_architecture_matched_pilot \
  --arms flat,factor,depth_identity,depth_split \
  --edge-budgets 29491,14746,6881 \
  --warmup-steps 120 --steps 80 --batch 16 --length 72 --eval-streams 256
/home/loganriggs/Coding/neuron_circuits/.venv/bin/python \
  language_task_suite/eval_sparse_architecture_causal_edits.py \
  --run-dir language_task_suite/sparse_architecture_matched_pilot \
  --arms flat,factor,depth_split --edge-budgets 29491,14746 \
  --nstreams 128 \
  --output language_task_suite/sparse_architecture_matched_pilot/causal_edits.json
/home/loganriggs/Coding/neuron_circuits/.venv/bin/python \
  language_task_suite/distill_sparse_depth_vs_factor.py \
  --output-dir language_task_suite/sparse_architecture_recovery_pilot \
  --arms flat,factor,depth_split \
  --recover-from-dir language_task_suite/sparse_architecture_matched_pilot \
  --recover-budget 29491 --recover-steps 300 \
  --batch 16 --length 72 --eval-streams 256
/home/loganriggs/Coding/neuron_circuits/.venv/bin/python \
  language_task_suite/eval_sparse_architecture_causal_edits.py \
  --run-dir language_task_suite/sparse_architecture_recovery_pilot \
  --arms flat,factor,depth_split --edge-budgets 29491 \
  --nstreams 128 \
  --output language_task_suite/sparse_architecture_recovery_pilot/causal_edits.json
```

