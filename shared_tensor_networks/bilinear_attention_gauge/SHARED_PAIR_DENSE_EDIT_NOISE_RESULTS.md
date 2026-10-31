# Edit-sparse gauges need accurate, diverse intervention responses

The [trained dense-token result](SHARED_PAIR_DENSE_EDIT_GAUGE_RESULTS.md)
fit an exact gauge from all 28 token-swap vectors and recovered the
known variables in ten models. This follow-up holds the **same rank-three
function and projector fixed** while corrupting only the observed
three-dimensional swap vectors used to fit its orthogonal gauge.

For each of three trained models, we ran ten independent noise draws
at each condition. Gaussian coordinate noise has standard deviation
`σ × median L2 norm of the 28 exact swap vectors`. The independent
condition draws fresh zero-mean noise for each pair and observation,
then **averages observations of each pair before fitting**. The fixed
pair-bias condition draws one error vector per pair and repeats it
unchanged; it is a controlled example of a systematic context or
measurement bias. Neither condition changes the model, its true
function, or the downstream task. Success below means post-fit cosine
above .98 with the known three variables, measured after fitting.

| Noise σ | Independent: 1 / 16 / 64 / 256 observations per pair | Fixed pair bias: 1 or 256 observations |
|---:|---|---:|
| 0 | 30 / 30 / 30 / 30 | 30 / 30 |
| .2 | 2 / 30 / 30 / 30 | 2 / 30 |
| .5 | 0 / 17 / 30 / 30 | 0 / 30 |
| 1.0 | 0 / 0 / 19 / 30 | 0 / 30 |

Each entry counts successes among **30 fits** (three models × ten noise
draws), not independent model replications. A quadratic-readout
concentration score shows the same broad transition: its one-node-per-
cell fraction exceeds .95 in all 30 trials at σ=.2 with 16 independent
observations, σ=.5 with 64, and σ=1 with 256. Fixed bias does not
improve with repeats by construction. The σ=.2 single-observation
result is a warning: exact all-pair geometry can be quite fragile to
noisy edit measurements even when every pair is represented.

This is **synthetic measurement noise**, not activation noise measured
in the 546M model. The gauge fitting is allowed to average repeated
measurements of the *same* known pair, which may be easier than
identifying matched contexts in real text. The noisy-fit sweep tests
variable alignment and learned quadratic-form concentration; it does
not rerun the 30,000-sentence pruning task for each noisy gauge.
Together with the [random-pair coverage sweep](SHARED_PAIR_DENSE_EDIT_GAUGE_RESULTS.md),
it suggests two separate requirements for using this prior at scale:
cover many distinct edit directions, and estimate each direction well
enough that context bias does not determine the L1 optimum. More draws
help the first only if they broaden support, and help the second only
when their errors are sufficiently independent.

Reproduce with `shared_pair_dense_edit_noise.py`; all condition scores
and seeds are in `shared_pair_dense_edit_noise.json`.
