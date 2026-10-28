# T2: spare capacity supports distributed robustness; capacity pressure changes the tradeoff

2026-10-28.24 trained bilinear toys under a [protocol fixed before outcomes](TOY_CAPACITY_PROTOCOL.md).
This is the first learning experiment in this series, not an attribution of language-model training.
All models have eight product channels and544 parameters. Three seeds per condition, all reported.

## What “too much task” means here

Sixteen token identities have either four shared target vectors or16 independent target vectors.
Targets have unit norm and covariance rank3 or15. An eight-channel decoder has output affine rank
at most8. The fine task therefore has an exact minimum squared reconstruction error of7/15≈.46667;
the coarse task can be fitted exactly. This does NOT mean sixteen-way classification is impossible
with eight channels. Reconstructing a prescribed geometry is harder than just separating labels.
Inputs include nuisance coordinates; token coordinates receive Gaussian noise at training sigma0
or.25. Test identities are familiar, but nuisance/noise draws are independent. There are no unseen
feature combinations in this first experiment. Label corruption and rare-feature robustness are untested.

Compare ordinary squared-error training with the same training plus.01 times the sum of squared
channel contribution norms. This favors smaller contributions, not a named “redundancy strategy.”
Models can discover different decompositions. Fixed2000steps, same paired initialization/data per
seed, no extension or outcome-based model selection. The clean fine-task runs reach their capacity
bound; their remaining reconstruction error is not simply an optimizer failure.

## Clean training: capacity changes the value of contribution spreading

Means across all three seeds. Deletion removes2/8 channels with16 fixed masks; no repair or rescaling.
“Participation” measures effective spread of channel magnitudes, not independent feature count.

| Targets | Penalty | Clean squared error | Clean accuracy | Channel/net energy | Participation | Deleted error | Deleted accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
|4 shared classes|0|~0|100%|.944|2.62|.262|89.20%|
|4 shared classes|.01|.00002|100%|.401|5.39|.121|100%|
|16 distinct vectors|0|.46667|95.83%|1.124|3.50|.651|74.87%|
|16 distinct vectors|.01|.46672|83.33%|1.000|4.01|.600|72.53%|

With spare capacity, the penalty changes the implementation substantially while preserving the
clean fit, and improves deletion robustness. Under output-rank pressure, it modestly improves
post-deletion reconstruction error but does not improve classification accuracy. Equal reconstruction
error is not equivalent clean classification performance. In the symmetric fine task, multiple
rank-eight approximations have the same optimal reconstruction error but different confusions.
Do not call one strategy uniformly better from these averages. Three seeds are descriptive,
not a population-level significance claim; per-seed confusion matrices are in the result JSON.

## Input noise is a separate robustness axis

For ordinary training (penalty0):

| Targets | Training sigma | Clean error | Accuracy at test sigma.25 | Error at test sigma.25 | Accuracy at sigma.5 |
|---|---:|---:|---:|---:|---:|
|4 classes|0|~0|81.97%|1.078|51.30%|
|4 classes|.25|.0976|93.46%|.302|59.15%|
|16 vectors|0|.4667|63.54%|1.014|29.43%|
|16 vectors|.25|.5002|71.06%|.596|42.87%|

Noise training improves noisy-input performance, but changes the clean function: clean vector
reconstruction worsens, even when coarse-task clean classification stays100%. On fine targets,
clean classification also falls from95.83% to72.92%. This is a tradeoff, not free robustness.

The penalty does not reliably improve input-noise robustness. With clean training and coarse
targets it raises sigma.25 accuracy from81.97% to84.93%, but lowers sigma.5 accuracy from51.30%
to49.54%. Under noisy training its effects are small. Thus channel-deletion robustness and input
ambiguity robustness cannot be combined into a single score.

## A useful mathematical explanation for the capacity interaction

For the matrix Y of bias-excluded outputs over inputs, each channel contributes a rank-one matrix
C_i (activation over examples times its decoder direction). With N channels:

    nuclear_norm(Y) <= sum_i Frobenius_norm(C_i)
                    <= sqrt(N * sum_i Frobenius_norm(C_i)^2).

Consequently the channel/net energy ratio is bounded below by
`nuclear_norm(Y)^2 / (N * Frobenius_norm(Y)^2)`.
For r equal singular values this is r/N. An exactly fitted rank-three target with eight channels
can approach3/8=.375 by spreading contributions. A rank-eight output with eight equal singular
values has a lower bound of1: there is no spare channel budget for the same kind of spreading.

Post-result analysis of the learned clean-training, penalized models gives observed ratio.401
against bound.376 for coarse targets, and1.000007 against bound1.000000 for fine targets.
This inequality is an exact constraint on the realized factorization, not an independent causal
explanation of optimizer dynamics. It helps explain why the same regularizer has different room
to act. It does not prove that width is the sole cause of every robustness difference.

## What we learned, and what still needs a test

The strongest result is conditional: when the required output rank leaves room, a contribution
penalty can buy distributed deletion robustness without materially sacrificing the fitted function.
When rank is saturated, the same intervention has less room and its benefits depend on the metric.
Input-noise training changes the learned function and addresses a different failure mode.

We have not compared forced pure clustering versus forced pure cancellation architectures. Nor
have we shown that a trained LLM selects its implementation for these reasons. The next causal
check should hold the fine16-vector task fixed and change only product width,8 versus16, with
paired penalties and seeds. The full target rank15 then fits, and the contribution-energy bound
has a known change. Use this to test the capacity explanation before adding label noise or task
families. Width increases parameter count intentionally as the capacity intervention; do not call
it an equal-budget strategy comparison. Preserve the original run budget and input-noise conditions,
report failures, and do not retune optimizer/penalty per width. Separately, held-out-combination
conjunction tasks remain necessary before conclusions about reusable compositional computation.

## Reproduction and validation

```bash
python train_capacity_toys.py --output validation/capacity_toys.json
python audit_capacity_toys.py
python analyze_capacity_geometry.py
```

The trainer writes24 states and clean evaluation samples to `/tmp/emlp_t2_models.pt`; the audits
expect that file. Parameter-only weights are also bundled as
[JSON](validation/capacity_toy_weights.json), keyed by result row. No LLM weights are included.
Independent NumPy replay covers all24 clean forwards and384 deletion evaluations, including exact
confusion counts; maximum forward/metric errors2.66e-7/6.86e-8. All capacity lower bounds verified
from target singular values. Noisy-input metrics are produced by the same scoring code but not
independently replayed in this audit. All seeds and fixed-step loss histories are retained.
[Results](validation/capacity_toys.json), [audit](validation/capacity_toys_audit.json),
[geometry bound](validation/capacity_geometry_bound.json).
