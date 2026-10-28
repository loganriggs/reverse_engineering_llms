# T3 fixed protocol: change capacity on the same fine target task

Frozen before T3 training outcomes, 2026-10-28. Reuse the12 width-eight fine-task runs from T2;
train12 width-sixteen counterparts with the same16 centered unit target vectors, online balanced
batch256, nuisance coordinates, training token noise0/.25, contribution penalties0/.01, seeds0/1/2,
Adam lr.01 and2000steps. No optimizer/penalty retuning or early-stopping selection. The initializer
keeps the same per-parameter Gaussian distribution(std.2); different tensor shapes mean cross-width
initial functions are not identical and their variance can differ. Paired seeds give matched data
streams, not a perfect initialization-controlled intervention.

Width8 has544parameters; width16 has1072. Changing parameter count is intentional. Output rank15
can fit at width16; width8 has minimum target vector MSE7/15. The wider system has only one channel
beyond target rank, not abundant spare capacity. At perfect fit its contribution-energy lower bound
is15/16=.9375, compared with3/8=.375 in T2's coarse-task width8 example. Thus do not expect unlimited
redundancy simply because the rank bottleneck is removed.

Use the same independent balanced1024-example evaluation draws at token noise0,.25,.5. Score
vector error, nearest-prototype accuracy and confusion matrices. Delete a fixed25% of channels:
16seeded masks remove4/16, compared with2/8 in the reused runs. Report absolute and intact-relative
damage with seed-level values. Changes in clean fit must accompany robustness comparisons.
Save all models and clean/noisy evaluation samples, verify independent forwards and confusion
counts. Retain optimization failures. Reused T2 outcomes are known; T3 is a planned follow-up,
not a blind replication. No unseen-identity, held-out-combination or label-noise claim.
