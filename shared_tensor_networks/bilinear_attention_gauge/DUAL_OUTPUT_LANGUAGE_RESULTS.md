# Necessary multi-head attention: metric choice, rank-one ceiling, and fidelity tradeoffs

The first trained toy language let either head solve almost the whole task.
`train_dual_output_language.py` strengthens it. A sentence contains an A bit,
a B bit, a filler, and a final query. The common query requests outputs
`(A,B)`; a 5%-frequency rare query requests `(B,A)`. Each head emits **one
scalar**, and a learned two-output linear readout combines them. The rare
`A0,B1` combination is absent during training. Both tokens, the rare query,
and other combinations are observed. Ten CPU seeds trained for 2,500 steps
all reach 100% joint accuracy on all 96 exhaustive sentences, including the
12 held-out rare-query sentences.

This time the ablation gate passes: deleting either head gives at most 75%
exhaustive joint accuracy across the ten seeds (typically 50–63%). A single
scalar head cannot linearly separate both independent output bits on every
combination, so the architecture makes a two-head computation necessary.

## Exact function and three measures

`analyze_dual_output_language.py` derives each head's query/key/value logit
table exactly from its weights. Adding pair contributions and the readout bias
reproduces direct model logits on all 96 sentences within `1.1e-14`. On the
entire role-valid two-query/six-key domain, QK1 has rank two even though its
physical width is four; the exact domain quotient precedes rank-one WTG or
full-environment fitting. This is not a gauge of the full width-four bond.

The comparison uses (a) corpus-free uniform pair weighting, (b) empirical
pair frequencies from 100,000 synthetic training-distribution sentences,
and (c) the full-sequence squared-logit environment from those sentences.
Two additional objectives mix the normalized sequence environment 50:50
with either uniform pair fidelity or a QK2-to-constant intervention
environment. Each fit is validated on 50,000 fresh training-law sentences
and tested on every exhaustive combination. An independent second 100,000
sentences estimates measure sampling noise. The sequence and pair
environments differ by at least **5.35×** that sampling noise across all
heads and seeds; whole-sequence co-occurrence changes the reduction metric.

After rank one per head, full-environment fitting retains 100% exhaustive
joint accuracy for **9/10 seeds under every measure**. Seed 2 is the
discriminator:

| Seed 2 fit measure | Exhaustive joint accuracy | Held-out rare accuracy | Validation BCE |
|---|---:|---:|---:|
| Uniform pair, weight-only | 87.5% | 0% | 0.00168 |
| Empirical pair | 75% | 0% | 0.01328 |
| Empirical sequence | 75% | 0% | 0.01728 |
| Half sequence, half uniform | 81.25% | 0% | 0.00555 |
| Half sequence, half QK2 intervention | 81.25% | 0% | 0.00570 |

The mixed objectives partly protect overall accuracy but do not rescue the
held-out rare combination at this fixed rank. The 9/10 success rate means
the seed-2 ordering is a **specific failure mode**, not evidence that
weight-only reduction generally beats data-aware reduction. More than
classification accuracy is needed to compare the successful seeds:
logit error and QK2-intervention error remain distinct outputs in each JSON.

## Is rank one intrinsically insufficient?

No. `dual_rank1_oracle.py` fits two rank-one QK1 matrices against labels for
**all 96 sentences**, explicitly including the held-out combination. On
seed 2 it reaches 100% joint accuracy. At a task-plus-function objective
weight of 1, its relative logit MSE to the original model is 0.1055;
the uniform-pair/FET solution has lower MSE 0.0820 but only 87.5%
accuracy. The oracle proves that this rank budget *can* support the task;
it is not a valid generalization result because it sees the held-out labels.

Training rank-one matrices only on observed combinations, with starts
selected on independent observed-distribution validation, gives a sharper
warning. For seed 2, task loss plus a small observed-sequence fidelity
weight (0.01) happens to recover the held-out rule at 100% accuracy, but
its relative exhaustive logit MSE is **0.430**. Its QK2-to-constant
intervention errors are **0.723 and 0.157** for the two heads, versus
**0.086 and 0.073** for the uniform-pair/FET fit. Stronger fidelity weights
restore more of the original function while losing that held-out rule.
Using a uniform function-fidelity term over all 96 combinations, with no
labels for the holdout, does not reliably rescue it in the tested sweep.
These are exploratory single-seed supervised fits, not a new best method.

## Legibility priors on the learned heads

`analyze_dual_sparsity.py` applies the second stage—native-bond L1,
WTG-relative bond L1, pair-response L1, and QK2-intervention fidelity—to
all ten trained models. Penalty weights are swept; each head may spend at most
0.02 additional relative pair-function MSE above its rank-one fidelity fit.
Candidates failing a `1e-4` gradient-norm gate are excluded. Selection uses
only the fitted function and the stated prior, never task labels or the
held-out rare combination.

The results are modest. Under the uniform metric, the mean effective number
of pair responses is **6.712** for rank-one fidelity alone and **6.598**
for the selected pair-response L1 fits (a 1.7% reduction). Mean QK2-constant
intervention error rises from **0.0106** to **0.0202**. Mean exhaustive joint
accuracy remains 98.75% only because nine seeds were already robust; the
difficult seed still loses its held-out rare rule. Under the empirical-pair
metric, pair-response L1 changes effective support **6.511→6.374** while
intervention error rises **0.0148→0.0262**. The intervention-specific prior
barely changes uniform intervention error at this rank and allowance.

These L1 penalties reduce a concentration proxy, not semantic complexity.
They do not produce a demonstrated human-legible mechanism, improve rare-rule
transfer, or rescue causal fidelity. Native-weight L1 is additionally
gauge-dependent. A better next test is a **class/group prior with known
provenance** and a broader intervention family, judged against the unsupervised
priors at matched function and task error. The generator's class labels must
remain outside the unsupervised conditions.

The distinction matters for the research objective: **task prediction,
faithful reconstruction, and causal transfer can rank the same rank-one
circuits differently**. A canonical gauge and a sparsity prior must be
judged on all three, with held-out labels excluded from model selection.
The next step is a class/group prior and a broader intervention panel on this
genuinely multi-head task, testing whether a named component predicts
held-out input and internal interventions. The full-vocabulary 546M
matrix-free experiment remains outstanding.

## Reproduce

Run on CPU 2 on this host to avoid the intermittent numerical-process
crashes seen on another core. The results use no GPU.

```bash
cd /home/loganriggs/Coding/reverse_engineering_llms/shared_tensor_networks/bilinear_attention_gauge
for seed in $(seq 0 9); do
  taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/loganriggs/Coding/neuron_circuits/.venv/bin/python train_dual_output_language.py \
    --steps 2500 --seed "$seed"
  taskset -c 2 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /home/loganriggs/Coding/neuron_circuits/.venv/bin/python analyze_dual_output_language.py \
    --seed "$seed" --n-fit 100000 --n-val 50000
done
```

The supervised capacity script has `--fit-domain exhaustive` for the
label-aware ceiling and `--fit-domain train` for fits that exclude held-out
labels. In train mode, candidate starts are selected using independent
training-law validation; the exhaustive set is used only for final scoring.
