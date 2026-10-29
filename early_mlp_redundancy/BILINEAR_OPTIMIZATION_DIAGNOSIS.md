# Why the bilinear warmup fails: exact replay and optimizer intervention

M83 and M84 are complete. Two instrumented replays exactly reproduce their original trajectories. Removing Adam's first-moment memory improves some final models but does not stabilize the sparse models. No causal-supervision calibration or downstream training has started.

## Replay findings

Shared_fixed71 was chosen posthoc as the largest final failure; dense71 was chosen as the sole full-training improvement. These are diagnostic choices, not model selection. Across both replays, all2,000batchlosses, final parameter tensors and finalAdamstates match the originals exactly. Additional measurements do not change the trajectory.

The rare position (cache1280) appears30times in29batches out of1,000. Shared_fixed71 has505updates that increase its error,504without the position in the batch. Sampling it reduces its error on28/29appearances. All10largest increases occur without it in the batch;6coincide with support changes. The rare-position error peaks around6.40million and finishes around3.82million.

Dense71 has no support gates, yet its rare-position error transiently reaches2.94million before finishing at5.54. Hard selection is therefore not necessary for this transient instability. Conversely, the shared model's108support changes and several coincident jumps do not establish routing as the sole cause. The logs show when error grows; they do not by themselves isolate why a particular update grows it.

## Controlled first-moment intervention

M84 sets Adam beta1=0 instead of0.9 while retaining beta2=.999,lr=.001,eps=1e-8, all8initial states,1,000updates,128-example batches, original sample streams, data and decoder renormalization. First-moment bias correction changes as required. All8final states are retained; no early stopping, clipping, data exclusion or loss-weight change occurs.

An independent audit verifies optimizer defaults/stepcounts, schedules, buffers, finalstate agreement with saved progress, and full-training NumPy reconstruction/gates. No heldout outcomes enter this test. Values are full-training MSE in fixed output-normalized units; the constant mean-write reference has MSE1.

| Kind | Seed | Original Adam | No first-moment memory |
|---|---:|---:|---:|
| dense | 71 | 0.769 | 1.045 |
| scalar | 71 | 21.970 | 5.166 |
| block | 71 | 30.897 | 4.476 |
| shared_fixed | 71 | 1021.762 | 56.076 |
| dense | 72 | 1.580 | 0.902 |
| scalar | 72 | 241.945 | 407.033 |
| block | 72 | 476.930 | 93.815 |
| shared_fixed | 72 | 94.294 | 466.138 |

Five of eight improve relative to originalAdam, but all six sparse variants remain much worse than initialization and the mean-write reference. Removing memory is therefore not sufficient to fix this setup. It also changes update magnitudes and directions; those were not controlled separately. Neither result establishes an optimal optimizer or says that shared nonlinear features cannot be learned.

## Theory-backed next test and stopping criterion

Bilinear outputs can amplify high input norms, and their parameter gradients can be correspondingly large. Under uniform minibatching, a position contributes with weight1/128when sampled and zero otherwise; under the full uniform objective its weight is1/4092on every update. This changes gradient variance and intermittency without changing the intended mean reconstruction objective. Adam's nonlinear normalization means averaging gradients first is not equivalent to averaging stochastic updates.

The next fixed comparison uses the full4,092-position gradient on every update, originalAdamsettings and all8originalstarts for1,000updates. It preserves all positions and the uniform objective, but increases example exposure and compute; no matched-compute superiority claim is permitted. Compare to both completed controls. This is one sampling-granularity intervention, not a batch-size/learning-rate search. Audit it, then stop this optimizer diagnostic series and explicitly decide readiness or an architectural/objective revision before additional fitting. Keep the broader goal—diverse causal supervision for reusable executable features—intact.

Artifacts: validation/bilinear_input_optimization_replay.json, bilinear_input_optimization_replay_analysis.json, bilinear_input_nomomentum_protocol.json, bilinear_input_nomomentum_results.json, bilinear_input_nomomentum_audit.json, and bilinear_input_nomomentum_error_diagnostic.json. Replay selection, original failures, all states and all data are preserved.
