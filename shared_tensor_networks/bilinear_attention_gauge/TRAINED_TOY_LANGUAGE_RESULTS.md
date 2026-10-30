# Trained two-head toy language: reduction metrics and a task-design failure

`train_toy_language.py` trains a one-layer, two-head bilinear-attention model.
Each four-token sentence contains an `A{0,1}` token, a `B{0,1}` token, a
filler, and a final `QA` or `QB` query. The label is the bit carried by the
queried class. `QB` occurs in 5% of training examples; `QB,A1,B1` is never
seen in training. The first three tokens are permuted. Three CPU seeds were
trained for 1,500 steps with batch size 512; all reach 100% on every one of
the 96 exhaustive sentences, including the 12 held-out rare-query sentences.

`analyze_toy_language.py` constructs each head's exact query-token/key-token
logit contribution from its weights. The sum of all pair contributions
reproduces the model's logits on every exhaustive sentence to max absolute
error `5.4e-15`. All role-valid pairs are included: two query tokens and six
key tokens. The original QK1 width is four, but its score matrix on this
*entire* role-valid domain has rank two. The analysis first takes that exact
domain quotient, then compares WTG truncation and full-environment rank-one
fitting. It is not a gauge of the full physical width-four bond.

Three fit measures are compared:

- **Uniform pair:** exact, corpus-free weighting of all 12 role-valid pairs.
- **Empirical pair:** pair-frequency weighting from 100,000 independently
  generated training-distribution sentences.
- **Empirical sequence:** squared logit error on those 100,000 whole
  sentences, retaining cross terms between the three key contributions.

An independent 100,000-sentence sample estimates environment sampling noise;
50,000 fresh sentences measure predictive loss. All fits are then tested on
the 96 exhaustive sentences, including the training-held-out combination.

## What changed when whole sequences were used

After normalizing each environment by the squared value of the original
identity bond, relative Frobenius differences between the empirical pair and
sequence environments were:

| Seed | Head 0 | Head 1 | Largest same-measure sampling difference |
|---:|---:|---:|---:|
| 0 | 0.0240 | 0.0277 | 0.0010 |
| 1 | 0.0885 | 0.0062 | 0.0020 |
| 2 | 0.0892 | 0.0408 | 0.0025 |

The difference is much larger than finite-sample variation. The first-layer
function is an exact sum of pair interactions, but its **sequence-level
squared error** includes products between contributions from different keys.
A pair-frequency table therefore does not determine the whole-sequence
environment. This is a concrete reason to test more than pair co-occurrence
when choosing a data-aware canonicalization metric. It does not imply that
the sequence metric is automatically better for interpretability. The current
fit reduces each head separately; a joint reduction would also include
cross-head error terms.

## Reduction and task outcomes

All three measure choices and both rank-one methods retain 100% exhaustive
classification accuracy across all three seeds. Logit fidelity and
confidence differ. Full-environment fitting is generally more faithful than
direct WTG truncation under a given environment, but neither the uniform nor
the data-aware measure consistently wins on all 96 sentences. For example,
seed 2's held-out rare-query CE is 0.000286 after uniform-pair/FET,
0.001957 after empirical-pair/FET, and 0.000067 after empirical-sequence/FET.
Those are small absolute losses and should not be read as a general ranking.

The crucial red team is **head ablation**. In seeds 0 and 1, removing either
head still gives 100% exhaustive accuracy. In seed 2, removing head 0 also
gives 100%, and removing head 1 gives 93.75%. Thus this trained task does not
force distinct reusable computations into different heads. Accuracy cannot
validate a multi-head sparse explanation here; the learned circuit is partly
redundant. The next synthetic task should require two independent outputs or
multi-step binding, verify that both heads are causally needed, then repeat
the metric × legibility-prior comparison on held-out combinations and
intervention types.

The current test is one layer, not the proposed two-layer grammar task or the
546M model. The full-head matrix-free calculation and real-corpus measure
remain open.

## Reproduce

On this host, several unconstrained numerical processes crashed on the same
CPU; pinning these small jobs to CPU 2 completed reliably. This is a runtime
workaround, not a modeling assumption. No GPU is used.

```bash
cd /home/loganriggs/Coding/reverse_engineering_llms/shared_tensor_networks/bilinear_attention_gauge
for seed in 0 1 2; do
  taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/loganriggs/Coding/neuron_circuits/.venv/bin/python train_toy_language.py \
    --steps 1500 --batch-size 512 --seed "$seed"
  taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/loganriggs/Coding/neuron_circuits/.venv/bin/python analyze_toy_language.py \
    --seed "$seed" --n-fit 100000 --n-val 50000
done
```

Each seed has a compact `.pt` checkpoint, a training JSON, and an analysis
JSON in this directory. The fitting and validation generators use different
fixed seeds; every reported full-rank score is checked against direct model
evaluation.
