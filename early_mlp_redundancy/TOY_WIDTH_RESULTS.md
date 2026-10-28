# T3: capacity helps fit; input denoising and channel robustness favor different implementations

2026-10-28. Twelve new width16 toys compared with the twelve existing width8 fine-target toys,
under a [protocol fixed before outcomes](TOY_WIDTH_PROTOCOL.md). Same16 target vectors, noises,
penalties, seeds, data streams, optimizer and2000-step budget. Width increases544→1072parameters;
this is a capacity intervention, not an equal-cost architecture comparison. Per-parameter initialization
is unchanged but initial functions differ with width. Three seeds per condition, all retained.

## Removing the rank bottleneck

The fine target covariance has rank15. Width8 cannot reach zero vector error; width16 can.
Clean-trained width16 models fit nearly exactly. Means across seeds:

| Width | Contribution penalty | Clean error | Clean accuracy | Error after25%channel deletion | Accuracy after deletion | Channel/net energy |
|---|---:|---:|---:|---:|---:|---:|
|8|0|.46667|95.83%|.65047|74.87%|1.124|
|8|.01|.46672|83.33%|.60005|72.53%|1.000|
|16|0|~0|100%|.36455|88.02%|1.236|
|16|.01|.00009|100%|.24003|97.92%|.939|

Within width16, the penalty improves deletion robustness at essentially unchanged clean fit.
Across widths, lower post-deletion error does not imply lower incremental damage: width8 starts
with a large irreducible error. For example, unpenalized absolute error increase is approximately
.184 at width8 versus.365 at width16. The larger model's intact function contains more information
to lose. Always report both intact and damaged performance.

At a perfectly reconstructed rank15 output and width16, the contribution-energy lower bound is
15/16=.9375. The penalized clean-trained ratio.939 approaches that constraint. This model has
only one channel beyond target rank, unlike the much roomier rank3/width8 coarse toy. Capacity
and regularization interact; removing the rank bottleneck does not provide unlimited redundancy.

## Noise robustness remains distinct

At width16, input-noise training improves accuracy under sigma.25 noise from90.56% to97.85%
without a contribution penalty, and from90.82% to97.82% with it. Clean classification remains100%,
but clean vector error rises to.0626/.0664. At larger sigma.5 noise the corresponding accuracies
are54.95%/54.88%, so robustness does not extend arbitrarily far.

Bigger models are not uniformly more stable in vector error: under sigma.5 noise, clean-trained
width16 has error6.99 versus4.18 at width8 (unpenalized), although its classification accuracy is
higher41.63% versus29.43%. The output magnitude/geometry and label decision tell different stories.

For noise-trained width16 models, the contribution penalty does not improve25%-deletion accuracy:
77.54% versus77.42%, despite improving deletion vector error. Again, there is no universal winner.

## Why might noise change the implementation? A post-result specificity check

The noise-trained models have lower clean-input contribution participation than clean-trained
ones. To distinguish actual class specialization from arbitrary concentration, compute each
channel's mean squared contribution per clean class. Sum each channel's maximum class energy,
then divide by total class/channel energy. This statistic ranges from1/16 for uniform activity
across classes to1 for channels individually confined to one class. No fitted assignment or
threshold is used. It is descriptive, not a necessity test.

| Width16 training | Penalty | Class-energy specificity | Contribution participation |
|---|---:|---:|---:|
|Clean|0|.502|6.24|
|Clean|.01|.464|7.68|
|Noisy|0|.865|1.83|
|Noisy|.01|.986|1.63|

Thus in this toy family, noise training produces much more class-specific contributions on clean
inputs. The token one-hot geometry and isotropic corruption are particular modeling choices;
this is not a general theorem that noise always causes localization, nor an explanation of LLMs.

## Does specialization imply a specific vulnerability?

A post-result exhaustive stress test zeros each of16 channels separately in every fixed width16
model, on clean examples. No retraining or model selection. All intact clean classifications are
correct. Results are averaged over seeds:

| Training | Penalty | Mean accuracy across single-channel deletions | Classes with any single-channel error (out of16) | Mean class accuracy under its own worst channel |
|---|---:|---:|---:|---:|
|Clean|0|98.05%|5.00|68.75%|
|Clean|.01|99.87%|.33|97.92%|
|Noisy|0|94.46%|14.00|13.83%|
|Noisy|.01|94.19%|15.00|7.00%|

The last column allows a different deleted channel for each class. It is a white-box targeted
stress statistic, NOT the accuracy after one common channel deletion. Indeed, the worst individual
channel globally leaves93.75% accuracy in the noisy-trained conditions. Local specialization creates
selective vulnerabilities that a pooled global score largely hides. This is a concrete tradeoff
between noisy-input performance and internal-channel robustness in these trained toys.

These analyses are outcome-informed follow-ups, not preregistered confirmation. Seed-level
matrices and formulas are retained. The effects do not establish a universal Pareto frontier or
prove the same tradeoff is unavoidable under a different training objective.

## Next bounded causal test

Ask whether robustness to both perturbations can be trained, rather than treating the tradeoff as
fundamental. Freeze a small follow-up on the same fine target: input noise.25 and contribution
penalty.01, widths16 and32, with versus without training-time25% channel dropout; seeds0/1/2,
same optimizer and budget. Reuse the existing width16/no-dropout runs and train nine new runs.
Define dropout as unrescaled product-channel removal (matching the intervention), draw fresh masks
per batch, and use all channels at evaluation. Record clean/noisy, random-deletion and class-targeted
results together. Width32 deliberately adds capacity; report the cost. Do not tune dropout or
penalty after results or call any remaining tradeoff unavoidable. Stop this toy grid after the
fixed comparison, then map the resulting diagnostic signatures back to the real-model evidence
and its gaps. Context conjunction/held-out-combination tasks remain untested, not silently solved.

## Reproduction and validation

```bash
python train_width_toys.py --output validation/width_toys.json
python audit_width_toys.py
python analyze_toy_specialization.py
python analyze_toy_single_deletions.py
python audit_toy_single_deletions.py
```

T3 samples/states are written to `/tmp/emlp_t3_models.pt`; specialization also expects T2's
`/tmp/emlp_t2_models.pt`. Parameter-only [weights](validation/width_toy_weights.json) are bundled.
Independent NumPy replay covers12 models×(three input-noise levels+16 deletion masks)=228
metrics/confusion evaluations; maximum forward/metric errors1.28e-6/4.60e-7. Signed-energy ratios
and nuclear-norm bounds checked. All192 additional single-channel confusion matrices independently
match Torch exactly. [Results](validation/width_toys.json), [audit](validation/width_toys_audit.json),
[specificity](validation/toy_specialization.json), [single-channel results](validation/toy_single_deletions.json),
[single-channel audit](validation/toy_single_deletions_audit.json).
