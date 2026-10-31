# Diverse-task attention reduction and exact-gauge comparison

## Question and design

Does a data-aware reduction preserve task behavior better than a weight-only
reduction, and does a legibility prior recover reusable variables at a fixed
function? We trained five independently initialized three-head bilinear-attention
models on eleven Boolean outputs derived from three count comparisons A/B/C.
The outputs include primitive bits, three XORs, two ANDs, majority,
exactly-one, and all-equal. Keys have variable counts; a rare query rotates
the primitive outputs. The rare-query combination A=0, B=1, C=0 is absent
from model training and validation but present in test. Each selected model
scored 100% on all eleven test outputs, including that combination.

The exact first-layer response is a 6×8 table: two query forms, three bit
logits, eight key tokens. Direct model output and this table plus the frozen
nonlinear readout agree to better than 1e-8. The table generally has rank six,
though its leading three singular directions dominate. At rank two or three,
we compare a uniform-table SVD (weight-only), a training-frequency weighted
fit, a half-uniform mixture, and two task-response fits initialized from
the uniform and empirical cuts. Task-response fitting sees only the first six
model outputs and uses disjoint train/validation sequences. The other five
outputs are whole-task holdouts **for the reduction**, though the underlying
model was trained on them. Therefore this is transfer of a reduction, not
unsupervised discovery of new tasks.

## Reduction results

Numbers below are means over five seeds on 20,000 fresh test sequences per
seed. Joint accuracy requires every output in the named task set to be right.
Relative output MSE compares cut logits with the native model's logits.

| Rank-two cut | Common/all fit-task joint accuracy | Common/all held-task joint accuracy | Rare-combination fit-task joint accuracy | Rare-combination held-task joint accuracy | All fit-output relative MSE | All held-output relative MSE |
|---|---:|---:|---:|---:|---:|---:|
| Uniform table | .587 | .597 | .169 | .196 | .399 | .407 |
| Empirical table | .488 | .526 | .074 | .159 | .404 | .413 |
| Half-uniform mixture | .543 | .568 | .159 | .168 | .405 | .416 |
| Task fit from uniform | .585 | .614 | .000 | .025 | .340 | .393 |
| Task fit from empirical | .528 | .585 | .000 | .041 | .353 | .376 |

The supervised response fit lowers ordinary fit-output error and, in some
settings, held-output error. It nevertheless destroys joint performance on
the rare combination. Uniform table fitting is the best of these rank-two
choices on rare fit-task joint accuracy, but it is still poor. This is a
capacity limit as well as an objective conflict: the rules depend on three
independent count variables, while the reduction has only two key directions.
At rank three, all methods score 100% on both task sets and the rare
combination in this controlled toy. Rank-three uniform-table cuts differ
from the native table by 1.43% relative Frobenius norm on average, so the
perfect task score does **not** mean exact weight recovery.

## Legibility at a fixed rank-three function

For each seed, freeze the rank-three uniform-table cut and rotate its
orthonormal key coordinates. These rotations change neither the reduced
table nor any prediction. We compare its SVD gauge with two priors:

* **Key L1:** minimize absolute entries of the key-coordinate dictionary;
  this uses the reduced table but no examples or task labels.
* **Fit-response L1:** minimize the L1 norm of task-logit sensitivities to
  the three key coordinates on fresh training-distribution contexts, using
  only the six designated fit outputs. The total L2 norm is invariant under
  the rotation. Five held outputs and test contexts are excluded from fitting.

| Gauge | Mean cosine with known A/B/C directions | Key dictionary L1 | Fit-output sensitivity L1 | Held-output sensitivity L1 |
|---|---:|---:|---:|---:|
| SVD | .899 | 5.857 | 133.11 | 139.10 |
| Key L1 | .9997 | 4.312 | 98.99 | 131.92 |
| Fit-response L1 | .9997 | 4.331 | 98.56 | 131.88 |
| Random rotation mean | .831 | 6.360 | 144.45 | 139.64 |

Known-variable alignment improves over SVD in all five seeds for either
prior. Key L1 also improves held-output sensitivity sparsity in all five
seeds; the response prior's extra held-output gain over key L1 is negligible
and changes sign across seeds. The exact function-gauge reconstruction check
passes below 1e-10 relative error. The sensitivity metric is a derivative
score, **not** a finite-intervention causal-fidelity score. It also does not
measure whether a human can name a component without seeing the toy rules.

## Interpretation and next discriminator

This test separates reduction from legibility. Rank two loses a needed
variable, and a sparse gauge cannot put it back. Rank three retains task
behavior, and a simple sparse-key prior exposes the designed variables.
This success relies on the toy's sparse token coding: the earlier
[`DENSE_CODED_SHARED_PAIR_RESULTS.md`](DENSE_CODED_SHARED_PAIR_RESULTS.md)
red team showed that key-entry L1 fails when every token contributes to
every variable. Thus the present result supports **conditional** recovery,
not a universal key-sparsity prior. The trained model's physical heads may
still implement a different decomposition than these function-level axes.

We ran the first dense-code red team below. Finite-intervention sparsity and
physical head recovery remain open. The 546M continuation should test
multiple task families and held-out edits at matched reduction fidelity;
earlier full-context results establish compression/response tradeoffs but no
stable human-readable output rule.

## Dense-token red team

We repeated the entire eleven-task training and reduction comparison with
eight key tokens coded by all three overlapping ±1 attributes. Every token
now contributes to every designed variable. Five newly trained models again
solve all eleven test outputs and the rare combination. Rank-three uniform
cuts retain 100% task accuracy, though their tables differ from native by
.79% relative Frobenius norm on average. Rank-two results remain sensitive
to the reduction metric:

| Rank-two cut, dense code | All fit-task joint accuracy | All held-task joint accuracy | Rare fit-task joint accuracy | Rare held-task joint accuracy |
|---|---:|---:|---:|---:|
| Uniform table | .570 | .587 | .000 | .063 |
| Empirical table | .526 | .542 | .391 | .378 |
| Half-uniform mixture | .556 | .578 | .000 | .022 |
| Task fit from uniform | .556 | .590 | .000 | .091 |
| Task fit from empirical | .355 | .432 | .000 | .003 |

Here the empirical table metric protects the rare composition better than
the uniform metric, the opposite of the sparse-token coding. A task-fit cut
can lower common output MSE yet collapse rare joint accuracy. Neither metric
universally wins, and rank two is still inadequate for the full task suite.

At the **same fixed rank-three cut**, the choice of coordinate prior matters
strongly. The table below compares SVD, key L1, and a response-sensitivity L1
gauge fitted on only the three XOR outputs A⊕B, B⊕C, A⊕C. Primitive A/B/C,
both ANDs, majority, exactly-one, and all-equal are whole-task holdouts for
the gauge. The sensitivity L1 number is a normalized aggregate derivative
score on held-out outputs and fresh test contexts; lower is sparser.

| Dense-code gauge | Mean cosine with known A/B/C directions | Held-output sensitivity L1 |
|---|---:|---:|
| SVD | .884 | 137.79 |
| Key L1 | .700 | 159.09 |
| XOR-response L1 | 1.000 | 102.56 |
| Random rotation mean | .829 | 149.44 |

The response gauge recovers the designed directions in all five seeds even
without primitive outputs in its fitting objective. A broader composite-only
fit and the original six-output fit give the same near-perfect alignment.
Held-output sensitivity L1 is lower than SVD and key L1 in every seed;
the exact-gauge reconstruction error is below 1e-15.
The key-L1 failure reproduces the coding dependence seen in the earlier
shared-pair red team. All three gauges compute the **identical rank-three
reduced function** to numerical precision. The response prior uses task-trained
model logits and a chosen set of supervised task channels; this is evidence
for a useful supervised prior and held-task transfer, not an unlabeled
discovery claim. Whether the same axes map to selectively manipulable modules
in the 546M model remains open.

## Fresh finite-axis check

We used 3,000 new dense-token contexts per seed, unseen in either gauge fit.
For each gauge axis, we subtracted a fraction of that axis's contribution to
the three bit logits, rescaling axes to the same RMS bit-logit dose. We then
measured the L1/L2 sparsity of **finite** logit changes on the eight outputs
held out from the XOR gauge fit. Lower is sparser.

| Fraction of matched axis dose | SVD | Key L1 | XOR-response L1 |
|---|---:|---:|---:|
| .05 | 109.46 | 123.65 | 83.23 |
| .25 | 114.21 | 129.91 | 85.34 |
| 1.00 | 161.60 | 162.95 | 164.51 |

The response prior wins at the two smaller doses in all five seeds. At a
full-axis deletion, its advantage disappears and it is slightly less sparse
on average than either baseline. The large edit strongly changes the input
to the frozen nonlinear readout; a derivative-based prior is therefore local.
These are interventions **inside the reduced toy function**, not a test that
the original physical attention heads admit the same independent edits.
The next test should fit finite-response objectives at declared doses and
challenge them with unseen doses and real token edits.

## Reproduction

Run `diverse_task_benchmark.py --seed 0` (repeat seeds 0–4), then
`diverse_task_gauge.py`. Add `--coding dense` to both for dense-token runs and
`--fit-mode xor` to the gauge script for the strictest task-family holdout.
The benchmark saves selected checkpoints, scores, and cut matrices in
`diverse_task_benchmark*_seed*.pt/.json`; the gauge script saves dictionaries
and per-seed scores in `diverse_task_gauge*.json`. Run `diverse_task_finite.py`
for the fresh finite-axis check. All use CPU only. The
native-table equality, rank constraints, and exact-gauge equality are
asserted in code.
