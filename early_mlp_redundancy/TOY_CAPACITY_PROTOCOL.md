# T2 fixed protocol: capacity, input ambiguity and implementation robustness

Frozen before training outcomes, 2026-10-28.24 runs: two target types × two training-noise levels
× two contribution penalties × three seeds. No hyperparameter search or early-stopping selection.

Input:16 token identities represented as one-hot vectors, eight independent unit-Gaussian nuisance
coordinates, and one constant coordinate. Model:25→8 bilinear products→16 outputs, plus output bias.
All models have identical architecture. Targets are unit-norm centered one-hot prototypes: four
classes (four tokens share each target) or16 independent targets. Target covariance ranks are3 and15.
The output lies in an affine subspace of dimension at most8. Thus the fine-target minimum population
squared vector error is at least7/15 on uniformly sampled identities; coarse targets have no such
rank obstruction. This is a vector-reconstruction capacity bound, NOT a classification bound:
an eight-dimensional code can still distinguish16 identities.

Training: online balanced256-example batches, fresh independent nuisance and token noise each step.
Token-coordinate Gaussian noise sigma0 or.25; labels remain the underlying clean identity/class.
Adam learning rate.01,2000steps, no weight decay, same random data and initialization per seed across
penalty conditions. Objective: mean squared vector error plus lambda × mean sum of squared channel
write norms; lambda0 or.01. This penalizes large contributions, not cancellation specifically.
Seeds0,1,2. Record loss at fixed500step intervals; do not extend failures silently.

Evaluation: fixed independent1024-example balanced batches at token-noise sigma0,.25,.5, same
nuisance/noise across paired models. Record squared vector error, nearest-target classification
accuracy and full confusion matrices. Clean/noisy inputs are new draws, not unseen identities or
unseen feature combinations. This first screen does not test rare labels, label noise, context
conjunctions or held-out combinations; those remain separate questions.

Implementation robustness on clean evaluation:16 fixed masks, each removes exactly2/8 product
channels without rescaling or retraining; preserve output bias. Record error/accuracy per mask and
relative-to-intact damage, never damage alone. Native geometry: channel/net energy ratio, signed
interference, effective participation and output covariance spectrum. Save trained tensors and
samples to enable independent forward replay. Report all seeds, including failed fits; compare
robustness jointly with clean fit. No causal claim that a measured geometry is a deliberately
selected strategy. LLM checkpoint evidence is not used to choose toy outcomes or tune parameters.
