# A downstream-aware attention reduction can improve a common behavior and erase a rare one

This test extends the [trained three-count language](THREE_COUNT_CANONICAL_RESULTS.md)
control to ask what changes when the first attention layer feeds a nonlinear
computation. Ten saved, independently trained three-head bilinear-attention
models already solve three count-comparison bits on variable-length token
sequences, including a rare whole-combination holdout. Their exact
query × key × output tables are known from weights. We append an **analytic**
bilinear readout `p = -logit[0] * logit[1]`; its sign is the XOR of the first
two model-output bits. The attention models were trained; this extra readout
was not. It is a controlled two-stage computation, **not** a trained
two-layer transformer or evidence about natural-language parity circuits.

The reduction has rank two in the shared key-token dimension of the
`6 × 8` combined table. Rank three is a capacity control. All rank-two fits
keep the original model's output bias. No task labels enter any reduction
fit: context objectives regress on the trained model's original logits and
its deterministic bilinear output. Each seed uses 20,000 unlabeled
training-law contexts, 5,000 validation-law contexts, and 30,000 independent
test contexts. Training and validation exclude the rare-query combination
`A=0, B=1, C=0`; each test seed has 518–595 such examples.
Two further 10,000-example test panels change the Poisson key-count rate
from .75 to .35 and 1.2 without refitting.

The candidate objectives are: exact uniform table SVD (weight-only),
frequency-weighted table least squares, their half mixture, natural-context
logit MSE, and natural-context logit MSE plus a normalized bilinear-output
MSE with varied coefficients. Further conditions add uniform table MSE to
the bilinear-output objective. Each context fit uses two starts, Adam, and
validation-objective checkpoint selection. All methods are compared at the
same rank; the rank-three control has a different budget.

| Reduction | Full-test parity accuracy | Rare holdout parity accuracy | Full-test three-bit joint accuracy | Uniform table MSE |
|---|---:|---:|---:|---:|
| Uniform table, rank 2 | .685 | .523 | .571 | .318 |
| Frequency table, rank 2 | .734 | .709 | .559 | .323 |
| Half uniform / frequency table, rank 2 | .680 | .683 | .557 | .320 |
| Natural-context logits, rank 2 | .706 | .627 | .583 | .324 |
| Natural logits + 0.05 bilinear, rank 2 | .896 | .158 | .565 | .332 |
| Natural logits + 0.1 bilinear, rank 2 | **.931** | **.000** | .529 | .340 |
| Natural logits + 1 bilinear, rank 2 | .927 | .000 | .504 | .341 |
| Natural logits + 1 bilinear + 20 uniform, rank 2 | .881 | .108 | .541 | .326 |
| Natural logits + 1 bilinear + 100 uniform, rank 2 | .749 | .458 | .601 | .319 |
| Uniform table, rank 3 | **1.000** | **1.000** | **1.000** | **.0004** |

These are means over the ten trained checkpoints. At coefficient 0.1, all
ten rank-two fits score **0%** on the withheld combination despite about
93% overall parity accuracy. Compared with frequency-table fitting, its
paired-seed gain in overall parity accuracy is +.196 (bootstrap 95%
interval +.110 to +.277), while its rare-holdout accuracy changes by
-.709 (interval -.998 to -.418). The rarity of the withheld slice means a
high overall score can coexist with its complete failure. Rank-three
weight-only reduction preserves both; the failure is tied to the constrained
rank and metric choice, not the original model's inability to perform the
task.

The cross-metric picture is sharper than a single accuracy comparison.
Rank-two nonlinear fitting cuts its **bilinear-output relative MSE** from
.574 under the frequency-table fit to .090 at coefficient 0.1, while its
uniform table error increases from .323 to .340. It buys common downstream
fidelity by changing the retained function, with no label supervision. A
very strong uniform-table penalty (100) partly restores rare behavior but
drops full-test parity accuracy from .927 to .749. At weight 20, rare
accuracy remains only .108. Thus the mixed objective traces a tradeoff;
it does not make both desiderata free at rank two.

The count-rate shifts preserve the qualitative result. At rates .35 and
1.2, the coefficient-0.1 fit reaches .931 and .929 overall parity accuracy
but **0** on the rare holdout in both panels; each panel contains 168–213
withheld examples per seed. The frequency-table fit reaches .743/.729
overall and .703/.710 on the rare slice. The rank-three control remains
perfect in both shifts. The failure is stable across these changes in key
count, though this is still one grammar and one withheld combination.

This result argues for keeping three gates separate when canonicalizing
attention: broad-domain function fidelity, expected downstream behavior,
and whole-task transfer. The data-aware downstream objective is useful for
one goal and actively misleading for another. A sparse or otherwise legible
gauge chosen *after* this rank-two reduction cannot recover the missing rare
behavior; changing the gauge preserves the already reduced function. The
next two-stage test should compare exact-gauge legibility priors at matched
reconstruction budgets on a genuinely trained nonlinear toy, then validate
selected components on held-out interventions before scaling the prior to
the 546M model.

An [exact-gauge follow-up](THREE_COUNT_NONLINEAR_GAUGE_RESULTS.md) now
compares key sparsity with context-edit sparsity on these fixed rank-two
functions. It finds that sparse keys can leave nearly all of the nonlinear
edit response in a cross-axis interaction, while a context-edit gauge
concentrates individual effects by mixing the clean A/B key directions that
key sparsity recovers.

The [plot](three_count_nonlinear_metric.png) shows individual seeds and
means. Reproduce with `three_count_nonlinear_metric.py` and
`three_count_nonlinear_plot.py`. Full matrices, validation traces, and
per-seed outcomes are saved in
`three_count_nonlinear_metric_s0-1-2-3-4-5-6-7-8-9.json`; the longer check is
`three_count_nonlinear_metric_s0-1-2.json`.

## Limits

Some context fits select the step cap at 1,200; these are local solutions,
not globally optimal rank-two functions. A longer 4,000-step, three-start
check on seeds 0–2 changed validation objectives only slightly and kept
the coefficient-0.1 rare-holdout accuracy at zero. The 0.05 coefficient is
particularly seed-sensitive. The analytic readout is intentionally simple:
it establishes a metric/coverage failure mode, not a general theorem that
downstream-aware reduction harms rare language behavior.
